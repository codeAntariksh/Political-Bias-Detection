"""
Evaluate a trained PoliticalBiasModel checkpoint on the held-out test split.

Usage:
    python -m src.evaluate --features data/features --checkpoint checkpoints/best_model.pt
"""

import argparse
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from src.data import AllSidesDataset
from src.data.preprocessing import LABEL_NAMES
from src.models import PoliticalBiasModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="data/features")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    issue = np.load(os.path.join(args.features, "issue_tokens.npy"))
    topic = np.load(os.path.join(args.features, "topic_tokens.npy"))
    title = np.load(os.path.join(args.features, "title_tokens.npy"))
    body = np.load(os.path.join(args.features, "body_tokens.npy"))
    labels = np.load(os.path.join(args.features, "labels.npy"))

    indices = np.arange(len(labels))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42, stratify=labels)

    full_dataset = AllSidesDataset(issue, topic, title, body, labels)
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=args.batch_size, shuffle=False)

    model = PoliticalBiasModel().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(
                batch["issue"].to(device),
                batch["topic"].to(device),
                batch["title"].to(device),
                batch["body"].to(device),
            )
            test_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            test_labels.extend(batch["label"].cpu().numpy())

    print(f"Test Accuracy : {accuracy_score(test_labels, test_preds):.4f}")
    print(f"Test Macro-F1 : {f1_score(test_labels, test_preds, average='macro', zero_division=0):.4f}\n")
    print(classification_report(test_labels, test_preds, target_names=LABEL_NAMES))


if __name__ == "__main__":
    main()
