# Political Bias Detection — Issue-Topic Guided Cross-Attention Framework

A deep learning system that classifies news articles as **Left / Center /
Right** by modeling how a story's *headline* and *body* are framed in the
context of its underlying **issue** and **topic** — instead of treating bias
detection as plain text classification.

Built on **DeBERTa-v3-base**, with a custom cross-attention + gated-fusion
head trained on the AllSides dataset.

```
Issue + Topic ──► Issue-Topic Context Encoder ──► Cross-Attention (Headline) ─┐
                                              └──► Cross-Attention (Body)    ──┼─► Gated Fusion ─► Classifier ─► Left / Center / Right
```

See [`docs/architecture.md`](docs/architecture.md) for the full design
rationale and diagram.

## Why this approach

Two outlets can report the same event with near-identical facts yet frame it
very differently — e.g. an immigration story framed around humanitarian
concerns vs. national security. That framing is contextual: it depends on
*what issue and topic* are being discussed. This project makes that context
explicit and lets the model learn, via cross-attention, which parts of the
headline and body matter given that context — rather than concatenating all
fields and hoping the classifier figures it out.

## Key design choices

- **Token-level features, not CLS pooling.** DeBERTa's `last_hidden_state`
  is kept as a full sequence per field (`issue`, `topic`, `title`, `body`),
  so cross-attention has real sequences to attend over instead of a single
  pooled vector.
- **Issue-Topic Context Encoder.** Issue and topic token sequences are
  concatenated and passed through a small self-attention encoder, producing
  a unified context representation used as the *query* downstream.
- **Dual cross-attention branches.** Separate stacks for headline and body,
  each queried by the issue-topic context — this is the main departure from
  a concatenate-and-classify baseline.
- **Learnable attention pooling.** Replaces mean pooling with a
  softmax-weighted sum so the model can emphasize the most diagnostic
  tokens.
- **Gated fusion.** A learned gate adaptively weights headline vs. body
  signal per article (some articles reveal bias mainly in the headline,
  others mainly in the body), with gate-collapse monitoring during training.
- **Class-weighted loss + differential learning rates.** DeBERTa's unfrozen
  top layers train at `5e-6`; task-specific modules train at `2e-4`.
  Cross-entropy is class-weighted to handle AllSides' label imbalance.

## Repository structure

```
├── src/
│   ├── data/
│   │   ├── preprocessing.py     # CSV loading, label encoding, TokenExtractor (DeBERTa)
│   │   └── dataset.py           # AllSidesDataset (PyTorch Dataset)
│   ├── models/
│   │   ├── issue_topic_encoder.py
│   │   ├── cross_attention.py
│   │   ├── pooling.py
│   │   ├── gated_fusion.py
│   │   ├── classifier.py
│   │   └── political_bias_model.py   # assembles the full model
│   ├── extract_features.py      # cache DeBERTa token sequences to disk
│   ├── train.py                 # training loop (early stopping, gate diagnostics)
│   └── evaluate.py               # test-set evaluation + classification report
├── configs/config.yaml          # hyperparameters
├── docs/
│   ├── architecture.md          # full design write-up
│   └── research_notes.md        # documented future-work direction
├── notebooks/political_bias_v3.ipynb  # original exploratory notebook
├── data/
│   ├── README.md                # dataset setup instructions
│   └── sample_allsides.csv      # 45-row sample for a quick smoke test
└── requirements.txt
```

### 🚀 Quickstart: Running Inference
 The script will automatically download the pre-trained `best_model.pt` weights from Hugging Face Hub repository—no manual file handling is required.

#### 1. Clone & Setup
```bash
# Clone the repository
git clone [https://github.com/codeAntariksh/Political-Bias-Detection.git](https://github.com/codeAntariksh/Political-Bias-Detection.git)
cd Political-Bias-Detection

# Install the required dependencies
pip install -r requirements.txt
## Usage

1. **Get the data.** Place the full AllSides CSV at `data/allsides.csv`
   (see [`data/README.md`](data/README.md)), or use the bundled
   `data/sample_allsides.csv` for a quick smoke test.

2. **Extract DeBERTa features** (one-time; cached to disk):
   ```bash
   python -m src.extract_features --csv data/allsides.csv --out data/features
   ```

3. **Train:**
   ```bash
   python -m src.train --features data/features --out checkpoints/best_model.pt
   ```

4. **Evaluate on the held-out test split:**
   ```bash
   python -m src.evaluate --features data/features --checkpoint checkpoints/best_model.pt
   ```

## Model I/O shapes (batch size B)

| Field  | Shape          |
|--------|----------------|
| issue  | `[B, 16, 768]` |
| topic  | `[B, 16, 768]` |
| title  | `[B, 32, 768]` |
| body   | `[B, 128, 768]`|
| logits | `[B, 3]`       |

## Tech stack

Python · PyTorch · Hugging Face Transformers (DeBERTa-v3-base) · scikit-learn · pandas / NumPy

## Status & future work

The core architecture (feature extraction → context encoding →
cross-attention → gated fusion → classification) is implemented end-to-end
and runnable via the scripts above. See
[`docs/research_notes.md`](docs/research_notes.md) for a documented
extension using framing-aware contrastive/triplet objectives, which is
noted as future work rather than shipped code.

## License

[MIT](LICENSE)
