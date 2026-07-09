import argparse
import sys
import os
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
from huggingface_hub import hf_hub_download

# --- Constants & Configuration ---
ROBERTA_MODEL = 'jayanta/roberta-news-bias'
LABEL_NAMES = ['Liberal(left)', 'Center', 'Conservative(right)']
MAX_NEWS_LEN = 192
MAX_CTX_LEN = 64

# REPLACE THIS WITH YOUR ACTUAL HUGGING FACE REPOSITORY ID
HF_REPO_ID = "dheeraj98kp/political-bias-detection-head"

# --- Custom Model Architecture ---
class LinearProjectionBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.2):
        super().__init__()
        h = hidden_dim[0] if isinstance(hidden_dim, list) else hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(h, output_dim),
            nn.LayerNorm(output_dim, eps=1e-6),
        )

    def forward(self, x):
        return self.projection(x)

class NewsFeatureExtractor(nn.Module):
    def __init__(self, model_name=ROBERTA_MODEL, freeze_layers=6):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        for i, layer in enumerate(self.roberta.encoder.layer):
            if i < freeze_layers:
                for p in layer.parameters():
                    p.requires_grad_(False)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state
        cls_emb = out.pooler_output
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_emb = torch.sum(last_hidden * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_emb = sum_emb / sum_mask
        combined = torch.cat([cls_emb, mean_emb], dim=-1)
        return combined

def context_aware_cross_modality_attention(query, value, ctx_emb, gate_lin):
    gate_input = torch.cat([query, ctx_emb], dim=-1)
    gate = torch.sigmoid(gate_lin(gate_input))
    return value + gate * ctx_emb

def compound_state(H, H_context, linear):
    g = torch.sigmoid(linear(torch.cat([H, H_context], dim=-1)))
    return g * H_context + (1.0 - g) * H

class ClassifierLayer(nn.Module):
    def __init__(self, input_dim=2304, num_classes=3, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-6),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    def forward(self, x):
        return self.head(x)

class EndToEndModel(nn.Module):
    def __init__(self, model_name=ROBERTA_MODEL, hidden_dim_proj=[512], output_dim_proj=256, fusion_dim=2304, num_classes=3, freeze_layers=12):
        super().__init__()
        self.news_encoder = NewsFeatureExtractor(model_name, freeze_layers)
        self.issue_backbone = NewsFeatureExtractor(model_name, freeze_layers=12)
        self.topic_backbone = NewsFeatureExtractor(model_name, freeze_layers=12)

        self.issue_adapter = nn.Sequential(nn.Linear(1536, output_dim_proj), nn.LayerNorm(output_dim_proj, eps=1e-6))
        self.topic_adapter = nn.Sequential(nn.Linear(1536, output_dim_proj), nn.LayerNorm(output_dim_proj, eps=1e-6))
        self.news_norm = nn.LayerNorm(1536, eps=1e-6)

        self.W_k = nn.Linear(1536, output_dim_proj)
        self.W_q = nn.Linear(1536, output_dim_proj)
        self.W_v = nn.Linear(1536, output_dim_proj)

        self.k_proj = LinearProjectionBlock(output_dim_proj, hidden_dim_proj, output_dim_proj)
        self.q_proj = LinearProjectionBlock(output_dim_proj, hidden_dim_proj, output_dim_proj)
        self.v_proj = LinearProjectionBlock(output_dim_proj, hidden_dim_proj, output_dim_proj)

        self.gate_issue_lin = nn.Linear(output_dim_proj * 2, 1)
        self.gate_topic_lin = nn.Linear(output_dim_proj * 2, 1)
        self.gate_issue = nn.Linear(2 * output_dim_proj, 1)
        self.gate_topic = nn.Linear(2 * output_dim_proj, 1)

        self.classifier = ClassifierLayer(fusion_dim, num_classes)

    def forward(self, batch):
        news_raw = self.news_encoder(batch['news_input_ids'], batch['news_attention_mask'])
        issue_raw = self.issue_backbone(batch['issue_input_ids'], batch['issue_attention_mask'])
        topic_raw = self.topic_backbone(batch['topic_input_ids'], batch['topic_attention_mask'])

        news = self.news_norm(news_raw)
        issue_emb = self.issue_adapter(issue_raw)
        topic_emb = self.topic_adapter(topic_raw)

        key = self.k_proj(self.W_k(news))
        query = self.q_proj(self.W_q(news))
        value = self.v_proj(self.W_v(news))

        issue_ctx = context_aware_cross_modality_attention(query, value, issue_emb, self.gate_issue_lin)
        topic_ctx = context_aware_cross_modality_attention(query, value, topic_emb, self.gate_topic_lin)

        h_issue = compound_state(key, issue_ctx, self.gate_issue)
        h_topic = compound_state(key, topic_ctx, self.gate_topic)

        fused = torch.cat([news, h_issue, h_topic, value], dim=-1)
        return self.classifier(fused)

# --- Inference Logic ---
def tokenize_input(tokenizer, text, max_length, device):
    enc = tokenizer(
        str(text), add_special_tokens=True, max_length=max_length,
        padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
    )
    return enc['input_ids'].to(device), enc['attention_mask'].to(device)

def load_model(checkpoint_path: str, device: torch.device) -> EndToEndModel:
    if not os.path.exists(checkpoint_path):
        print(f"\n[Info] Local checkpoint not found at '{checkpoint_path}'.")
        print(f"Downloading weights automatically from Hugging Face Hub ({HF_REPO_ID})...")
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            
            # Download directly to the target path
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename="best_model.pt",
                local_dir=os.path.dirname(checkpoint_path)
            )
            print("Download complete!\n")
        except Exception as e:
            print(f"\n[ERROR] Could not download from Hugging Face: {e}")
            print("Please ensure your HF_REPO_ID is correct and public.")
            sys.exit(1)
            
    print("Initializing RoBERTa architecture...")
    model = EndToEndModel().to(device)
    
    try:
        print("Loading state dict...")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        print(f"\n[ERROR] Failed to load checkpoint: {e}")
        sys.exit(1)
        
    return model

def main():
    parser = argparse.ArgumentParser(description="Predict political bias with RoBERTa Model.")
    parser.add_argument("--checkpoint", type=str, default="content/best_model.pt", help="Path to checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading RoBERTa Tokenizers...")
    tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL)

    print(f"Looking for trained RoBERTa model at '{args.checkpoint}'...")
    model = load_model(args.checkpoint, device)

    print("\n=== RoBERTa Political Bias Detection ===")
    issue = input("Issue      : ").strip()
    topic = input("Topic      : ").strip()
    headline = input("Headline   : ").strip()
    print("News Body  : (paste full body, then press Enter)")
    news_body = input().strip()

    if not all([issue, topic, headline, news_body]):
        print("\n[ERROR] All fields are required.")
        sys.exit(1)

    # Format text same as training
    news_txt = ' '.join(f"{headline}. {news_body}".strip().split()[:MAX_NEWS_LEN])

    n_ids, n_mask = tokenize_input(tokenizer, news_txt, MAX_NEWS_LEN, device)
    i_ids, i_mask = tokenize_input(tokenizer, issue, MAX_CTX_LEN, device)
    t_ids, t_mask = tokenize_input(tokenizer, topic, MAX_CTX_LEN, device)

    batch = {
        'news_input_ids': n_ids, 'news_attention_mask': n_mask,
        'issue_input_ids': i_ids, 'issue_attention_mask': i_mask,
        'topic_input_ids': t_ids, 'topic_attention_mask': t_mask
    }

    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
    pred_idx = int(probs.argmax())
    label = LABEL_NAMES[pred_idx]

    print("\n----------- Prediction -----------")
    print(f"Predicted Bias : {label}")
    print("Class Probabilities:")
    for name, p in zip(LABEL_NAMES, probs):
        marker = "  <-- predicted" if name == label else ""
        print(f"  {name:<12}: {p * 100:6.2f}%{marker}")
    print("-----------------------------------")

if __name__ == "__main__":
    main()