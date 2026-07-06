import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """
    Adaptively fuses headline and body embeddings per-article, since some
    articles reveal bias mainly in the headline and others mainly in the body.

    gate  g = sigmoid(W2 . GELU(W1 . [H; B]))
    fused F = g * H + (1 - g) * B
    """

    def __init__(self, hidden_dim=768, dropout=0.10):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        # Diagnostics used during training to catch gate collapse
        # (gate_std < 0.05 means the gate has stopped discriminating).
        self.last_gate_mean = None
        self.last_gate_std = None

    def forward(self, headline_emb, body_emb):
        combined = torch.cat([headline_emb, body_emb], dim=-1)  # [B, 1536]
        gate = self.gate_network(combined)                       # [B, 768]
        with torch.no_grad():
            self.last_gate_mean = gate.mean().item()
            self.last_gate_std = gate.std().item()
        fused = gate * headline_emb + (1.0 - gate) * body_emb
        return fused
