# Data

This project uses the **AllSides** news-bias dataset (article title, body,
issue, topic, and a Left/Center/Right label).

The raw CSV (`allsides.csv`, ~9MB) is **not committed** to this repository
(see `.gitignore`) to keep the repo lightweight and avoid redistributing a
third-party dataset. To run the pipeline:

1. Place your copy of the dataset at `data/allsides.csv` with at least these
   columns: `Title`, `News_Body`, `issue`, `topic`, `Label`.
2. Run feature extraction to cache DeBERTa token sequences:
   ```bash
   python -m src.extract_features --csv data/allsides.csv --out data/features
   ```
   This writes `issue_tokens.npy`, `topic_tokens.npy`, `title_tokens.npy`,
   `body_tokens.npy`, and `labels.npy` into `data/features/` (also
   git-ignored — regenerate rather than commit).

If you don't have access to the original AllSides scrape, search for
"AllSides news bias dataset" — several public mirrors exist on Kaggle and
GitHub with this schema.
