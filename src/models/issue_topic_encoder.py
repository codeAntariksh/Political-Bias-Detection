import torch
import torch.nn as nn


class IssueTopicContextEncoder(nn.Module):
    """
    Fuses issue and topic token sequences into a single contextual
    representation via self-attention, instead of treating them as two
    independent pooled vectors.

    Input:
        issue_tokens : [B, Li, 768]
        topic_tokens : [B, Lt, 768]

    Output:
        issue_topic_context : [B, Li+Lt, 768]
    """

    def __init__(self, d_model=768, num_heads=4, ff_dim=1536, num_layers=2, dropout=0.10):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, issue_tokens, topic_tokens):
        x = torch.cat([issue_tokens, topic_tokens], dim=1)  # [B, Li+Lt, 768]
        return self.encoder(x)
