import torch.nn as nn


class CrossAttentionEncoderBlock(nn.Module):
    """
    A single cross-attention layer: the issue-topic context sequence queries
    a headline or body token sequence, so the model learns which parts of
    the headline/body matter given the specific issue-topic being discussed.

    Query     : issue_topic_context  [B, Li+Lt, 768]
    Key/Value : headline or body     [B, Lh/Lb, 768]
    """

    def __init__(self, d_model=768, num_heads=8, ff_dim=3072, dropout=0.10):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        # Kept for interpretability (e.g. attention-weight visualization).
        self.last_attn_weights = None

    def forward(self, query, key_value):
        attn_out, attn_weights = self.cross_attention(query=query, key=key_value, value=key_value)
        self.last_attn_weights = attn_weights.detach()

        x = self.norm1(query + self.dropout(attn_out))
        output = self.norm2(x + self.dropout(self.ffn(x)))
        return output, attn_weights


class CrossAttentionEncoder(nn.Module):
    """Stack of CrossAttentionEncoderBlocks. Output: [B, Li+Lt, 768]."""

    def __init__(self, num_layers=4, d_model=768, num_heads=8, ff_dim=3072, dropout=0.10):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionEncoderBlock(d_model=d_model, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, query, key_value):
        x = query
        for layer in self.layers:
            x, _ = layer(x, key_value)
        return x

    def get_last_attn_weights(self):
        return self.layers[-1].last_attn_weights
