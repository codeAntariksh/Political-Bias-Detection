"""PyTorch Dataset wrapping pre-extracted DeBERTa token sequences."""

import torch
from torch.utils.data import Dataset


class AllSidesDataset(Dataset):
    """
    Stores per-field token sequences [L, 768] (not pooled vectors), so the
    DataLoader can collate them into [B, L, 768] batches for cross-attention.
    """

    def __init__(self, issue, topic, title, body, labels):
        self.issue = torch.FloatTensor(issue)   # [N, 16,  768]
        self.topic = torch.FloatTensor(topic)   # [N, 16,  768]
        self.title = torch.FloatTensor(title)   # [N, 32,  768]
        self.body = torch.FloatTensor(body)     # [N, 128, 768]
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "issue": self.issue[idx],
            "topic": self.topic[idx],
            "title": self.title[idx],
            "body": self.body[idx],
            "label": self.labels[idx],
        }
