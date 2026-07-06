"""
Data loading, label encoding, and DeBERTa token-sequence extraction
for the AllSides political-bias dataset.
"""

import numpy as np
import pandas as pd
import torch
from transformers import DebertaV2Model, DebertaV2Tokenizer

LABEL_MAP = {
    "liberal": 0,
    "center": 1,
    "conservative": 2,
}

LABEL_NAMES = ["Left", "Center", "Right"]


def load_allsides(csv_path: str) -> pd.DataFrame:
    """Load the AllSides CSV, drop unused columns, and drop rows with NaNs."""
    df = pd.read_csv(csv_path)
    df = df.drop(["roundup", "Stance"], axis=1, errors="ignore")
    df = df.dropna()
    return df.reset_index(drop=True)


def encode_labels(df: pd.DataFrame, label_col: str = "Label"):
    """Filter to known labels and map them to integer class ids."""
    df = df[df[label_col].isin(LABEL_MAP.keys())].reset_index(drop=True)
    labels = df[label_col].map(LABEL_MAP).values
    return df, labels


class TokenExtractor:
    """
    Extracts token-level hidden states (not CLS-pooled embeddings) from
    DeBERTa-v3-base for each text field. Attention modules downstream need
    real sequences to attend over, so pooling happens later in the model,
    not here.

    Returns last_hidden_state tensors:
      issue   -> [N, 16,  768]
      topic   -> [N, 16,  768]
      Title   -> [N, 32,  768]
      News_Body -> [N, 128, 768]
    """

    MAX_LENGTHS = {
        "issue": 16,
        "topic": 16,
        "Title": 32,
        "News_Body": 128,
    }

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        batch_size: int = 16,
        unfreeze_top_layers: int = 4,
        device: torch.device = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
        self.model = DebertaV2Model.from_pretrained(model_name).to(self.device)

        # Freeze everything, then unfreeze the top-N transformer layers so
        # they can be fine-tuned with a small differential learning rate.
        for param in self.model.parameters():
            param.requires_grad = False

        encoder_layers = self.model.encoder.layer
        num_layers = len(encoder_layers)
        for layer in encoder_layers[num_layers - unfreeze_top_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        if hasattr(self.model.encoder, "rel_embeddings"):
            for param in self.model.encoder.rel_embeddings.parameters():
                param.requires_grad = True

    def _extract_tokens(self, texts, max_length):
        all_hidden = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self.model(**inputs)
                hidden = outputs.last_hidden_state
                all_hidden.append(hidden.cpu().numpy())
        return np.vstack(all_hidden)

    def extract(self, df: pd.DataFrame, columns=("issue", "topic", "Title", "News_Body")):
        results = {}
        for col in columns:
            max_len = self.MAX_LENGTHS[col]
            texts = df[col].fillna("").astype(str).tolist()
            results[col] = self._extract_tokens(texts, max_len)
        return results

    def get_trainable_params(self):
        """Unfrozen DeBERTa params, used as a separate optimizer group with a lower LR."""
        return [p for p in self.model.parameters() if p.requires_grad]
