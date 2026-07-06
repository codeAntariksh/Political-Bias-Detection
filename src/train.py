"""
Train PoliticalBiasModel on cached DeBERTa token sequences.

Usage:
    python -m src.train --features data/features --out checkpoints/best_model.pt
"""

import argparse
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup

from src.data import AllSidesDataset, TokenExtractor
from src.models import PoliticalBiasModel


def load_features(features_dir):
    issue = np.load(os.path.join(features_dir, "issue_tokens.npy"))
    topic = np.load(os.path.join(features_dir, "topic_tokens.npy"))
    title = np.load(os.path.join(features_dir, "title_tokens.npy"))
    body = np.load(os.path.join(features_dir, "body_tokens.npy"))
    labels = np.load(os.path.join(features_dir, "labels.npy"))
    return issue, topic, title, body, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="data/features")
    parser.add_argument("--out", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--deberta-lr", type=float, default=5e-6)
    parser.add_argument("--heads-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    issue, topic, title, body, labels = load_features(args.features)

    indices = np.arange(len(labels))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.15, random_state=42, stratify=labels[train_val_idx])

    full_dataset = AllSidesDataset(issue, topic, title, body, labels)
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=args.batch_size, shuffle=False)

    # Only needed here for its still-trainable DeBERTa top layers (differential LR group).
    extractor = TokenExtractor(batch_size=args.batch_size)
    model = PoliticalBiasModel().to(device)

    class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1, 2]), y=labels[train_idx])
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = CrossEntropyLoss(weight=class_weights_tensor)

    deberta_params = extractor.get_trainable_params()
    head_params = list(model.parameters())
    optimizer = AdamW(
        [
            {"params": deberta_params, "lr": args.deberta_lr},
            {"params": head_params, "lr": args.heads_lr},
        ],
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.10 * total_steps), num_training_steps=total_steps
    )

    best_val_macro_f1 = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        extractor.model.train()
        train_loss, train_preds, train_labels = 0.0, [], []

        for batch in train_loader:
            issue_b = batch["issue"].to(device)
            topic_b = batch["topic"].to(device)
            title_b = batch["title"].to(device)
            body_b = batch["body"].to(device)
            labels_b = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(issue_b, topic_b, title_b, body_b)
            loss = criterion(logits, labels_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(deberta_params) + list(head_params), max_norm=args.grad_clip)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            train_labels.extend(labels_b.cpu().numpy())

        model.eval()
        extractor.model.eval()
        val_loss, val_preds, val_labels = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                issue_b = batch["issue"].to(device)
                topic_b = batch["topic"].to(device)
                title_b = batch["title"].to(device)
                body_b = batch["body"].to(device)
                labels_b = batch["label"].to(device)

                logits = model(issue_b, topic_b, title_b, body_b)
                loss = criterion(logits, labels_b)
                val_loss += loss.item()
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels.extend(labels_b.cpu().numpy())

        val_macro_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(val_labels, val_preds)

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            patience_counter = 0
            torch.save(model.state_dict(), args.out)
            saved = "saved"
        else:
            patience_counter += 1
            saved = f"patience {patience_counter}/{args.patience}"

        print(
            f"Epoch [{epoch+1:02d}/{args.epochs}] "
            f"train_loss={train_loss/len(train_loader):.4f} "
            f"val_loss={val_loss/len(val_loader):.4f} val_acc={val_acc:.4f} "
            f"val_macro_f1={val_macro_f1:.4f} ({saved})"
        )

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Best validation macro-F1: {best_val_macro_f1:.4f}")


if __name__ == "__main__":
    main()
