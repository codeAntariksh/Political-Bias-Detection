import torch.nn as nn

from .issue_topic_encoder import IssueTopicContextEncoder
from .cross_attention import CrossAttentionEncoder
from .pooling import AttentionPooling
from .gated_fusion import GatedFusion
from .classifier import BiasClassifier


class PoliticalBiasModel(nn.Module):
    """
    Issue-Topic Guided Cross-Attention Framework for political bias detection.

        issue_tokens [B,16,768]  --+
                                    +--> IssueTopicContextEncoder --> context [B,32,768]
        topic_tokens [B,16,768]  --+
                                                    |
                          +-------------------------+-------------------------+
                          |                                                   |
        title_tokens [B,32,768] -> CrossAttentionEncoder(Q=context, KV=title) |
        body_tokens [B,128,768] --------------------------------------------> CrossAttentionEncoder(Q=context, KV=body)
                          |                                                   |
                  AttentionPooling                                    AttentionPooling
                          |                                                   |
                headline_emb [B,768]                                 body_emb [B,768]
                          +-------------------------+-------------------------+
                                                    |
                                              GatedFusion [B,768]
                                                    |
                                              BiasClassifier
                                                    |
                                              logits [B, 3]
    """

    def __init__(
        self,
        d_model=768,
        ctx_num_heads=4,
        ctx_ff_dim=1536,
        num_context_layers=2,
        cross_num_heads=8,
        cross_ff_dim=3072,
        num_cross_layers=4,
        dropout=0.10,
        num_classes=3,
    ):
        super().__init__()

        self.issue_topic_encoder = IssueTopicContextEncoder(
            d_model=d_model, num_heads=ctx_num_heads,
            ff_dim=ctx_ff_dim, num_layers=num_context_layers, dropout=dropout,
        )

        self.headline_encoder = CrossAttentionEncoder(
            num_layers=num_cross_layers, d_model=d_model,
            num_heads=cross_num_heads, ff_dim=cross_ff_dim, dropout=dropout,
        )

        self.body_encoder = CrossAttentionEncoder(
            num_layers=num_cross_layers, d_model=d_model,
            num_heads=cross_num_heads, ff_dim=cross_ff_dim, dropout=dropout,
        )

        self.headline_pool = AttentionPooling(d_model)
        self.body_pool = AttentionPooling(d_model)

        self.fusion = GatedFusion(hidden_dim=d_model, dropout=dropout)
        self.classifier = BiasClassifier(input_dim=d_model, num_classes=num_classes)

    def forward(self, issue, topic, title, body):
        # issue: [B,16,768]  topic: [B,16,768]  title: [B,32,768]  body: [B,128,768]
        issue_topic_ctx = self.issue_topic_encoder(issue, topic)          # [B,32,768]

        headline_seq = self.headline_encoder(issue_topic_ctx, title)      # [B,32,768]
        body_seq = self.body_encoder(issue_topic_ctx, body)               # [B,32,768]

        headline_emb = self.headline_pool(headline_seq)                  # [B,768]
        body_emb = self.body_pool(body_seq)                              # [B,768]

        fused = self.fusion(headline_emb, body_emb)                      # [B,768]
        logits = self.classifier(fused)                                  # [B,3]
        return logits
