"""
Extract and cache DeBERTa token sequences for the AllSides dataset.

Usage:
    python -m src.extract_features --csv data/allsides.csv --out data/features
"""

import argparse
import os

import numpy as np

from src.data import TokenExtractor, encode_labels, load_allsides


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/allsides.csv")
    parser.add_argument("--out", type=str, default="data/features")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = load_allsides(args.csv)
    df, labels = encode_labels(df)
    print(f"Loaded {len(df)} labeled rows.")

    extractor = TokenExtractor(batch_size=args.batch_size)
    token_seqs = extractor.extract(df)

    np.save(os.path.join(args.out, "issue_tokens.npy"), token_seqs["issue"])
    np.save(os.path.join(args.out, "topic_tokens.npy"), token_seqs["topic"])
    np.save(os.path.join(args.out, "title_tokens.npy"), token_seqs["Title"])
    np.save(os.path.join(args.out, "body_tokens.npy"), token_seqs["News_Body"])
    np.save(os.path.join(args.out, "labels.npy"), labels)

    print(f"Saved token sequences to {args.out}/")


if __name__ == "__main__":
    main()
