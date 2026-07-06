import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """
    Learnable attention pooling: [B, N, 768] -> [B, 768].

    score_i = Linear(768 -> 1)
    alpha   = softmax(score)          # [B, N, 1]
    out     = sum(alpha_i * h_i)      # [B, 768]

    Lets the model weight the most informative sequence positions instead
    of averaging all positions uniformly.
    """

    def __init__(self, d_model=768):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x):
        scores = self.score(x)                      # [B, N, 1]
        weights = torch.softmax(scores, dim=1)       # [B, N, 1]
        pooled = (weights * x).sum(dim=1)             # [B, 768]
        return pooled
