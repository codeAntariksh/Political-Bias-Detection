# Political Bias Detection

A deep learning system that classifies news articles as **Liberal (Left) /
Center / Conservative (Right)** by reasoning about a story's *headline +
body* jointly with its underlying **issue** and **topic** — instead of
treating bias detection as plain text classification.

> **Fastest way to try it:** clone the repo, `pip install -r requirements.txt`,
> then run `python predict.py`. The trained model weights are hosted on
> Hugging Face Hub and are **downloaded automatically on first run** — no
> manual file downloads, no dataset, no GPU setup required.

## What this project does

Two articles can report the same facts but frame them very differently
depending on context (e.g. a healthcare story framed around "protecting
families" vs. "government overreach"). This project feeds the model that
context explicitly, so it learns which parts of the headline and body are
diagnostic of bias *given* the issue/topic, rather than judging the text in
isolation.

**Inputs** (four fields per article):
| Field       | Description                              |
|-------------|-------------------------------------------|
| `Issue`     | Broad issue area, e.g. `Healthcare`        |
| `Topic`     | Narrower sub-topic, e.g. `Drug Pricing`    |
| `Headline`  | The article's title                        |
| `News Body` | The full article text                      |

**Output:** a predicted class — `Liberal(left)`, `Center`, or
`Conservative(right)` — plus a softmax confidence score for all three
classes.

## Model architecture (currently deployed — used by `predict.py`)

The shipped model is a **multi-encoder RoBERTa architecture with gated
cross-modality fusion**, built on top of [`jayanta/roberta-news-bias`](https://huggingface.co/jayanta/roberta-news-bias)
as the pretrained backbone:

```
Headline + Body ──► News Encoder (RoBERTa)   ──┐
Issue            ──► Issue Encoder (RoBERTa)  ──┼─► Gated Cross-Modality Fusion ──► Classifier ──► Liberal / Center / Conservative
Topic            ──► Topic Encoder (RoBERTa)  ──┘
```

- **Three RoBERTa encoders** — one for the combined headline+body text, and
  one each for issue and topic — each producing a pooled embedding (CLS +
  mean-pooled tokens).
- **Gated cross-modality attention** — the news embedding is projected into
  query/key/value vectors, and issue/topic embeddings are blended in via a
  learned sigmoid gate rather than simple concatenation, so the model can
  adaptively weight how much the issue/topic context should influence the
  final representation for a given article.
- **Fusion + classification head** — the news, gated-issue, gated-topic, and
  value representations are concatenated (2304-d) and passed through a
  GELU/LayerNorm/Dropout classifier head to produce 3-class logits.

**Reported performance** (held-out test split, 917 articles, balanced
~305/class): **55.5% accuracy**, macro-F1 ≈ 0.55. Full per-class
precision/recall/F1 is in [`notebooks/roberta-news-bias-detection.ipynb`](notebooks/roberta-news-bias-detection.ipynb).

> **Note on `src/` and the original notebook:** this repo also contains an
> earlier, separate architecture (`src/` package, `configs/config.yaml`,
> [`docs/architecture.md`](docs/architecture.md), `notebooks/political_bias_v3.ipynb`) —
> a DeBERTa-v3 cross-attention + gated-fusion design. That pipeline
> (`extract_features.py` → `train.py` → `evaluate.py`) is a self-contained,
> earlier iteration of this idea and is **not** what `predict.py` currently
> runs. `predict.py` uses the newer, self-contained RoBERTa architecture
> defined directly inside it. Treat `src/` as the original research
> implementation, and `predict.py` as the current, working, easiest way to
> get a prediction end-to-end.

## Repository structure

```
├── predict.py                    # ⭐ run this — interactive CLI inference, auto-downloads weights from HF Hub
├── src/                           # original DeBERTa cross-attention framework (research implementation)
│   ├── data/
│   ├── models/
│   ├── extract_features.py
│   ├── train.py
│   └── evaluate.py
├── configs/config.yaml           # hyperparameters for the src/ pipeline
├── docs/
│   ├── architecture.md           # design write-up for the src/ (DeBERTa) framework
│   └── research_notes.md         # documented future-work direction
├── notebooks/
│   ├── political_bias_v3.ipynb            # original DeBERTa cross-attention exploration
│   └── roberta-news-bias-detection.ipynb  # training notebook for the current deployed RoBERTa model
├── data/
│   ├── README.md                 # dataset setup instructions
│   └── sample_allsides.csv       # 45-row sample for a quick smoke test
└── requirements.txt
```

### 🚀 Quickstart: Running Inference
 The script will automatically download the pre-trained `best_model.pt` weights from Hugging Face Hub repository—no manual file handling is required.

#### 1. Clone & Setup
```bash
# Clone the repository
git clone [https://github.com/codeAntariksh/Political-Bias-Detection.git](https://github.com/codeAntariksh/Political-Bias-Detection.git)
cd Political-Bias-Detection

That's it — no dataset download and no checkpoint download needed just to
try a prediction (see below).

## Usage

### Run inference (recommended — works out of the box)

```bash
python predict.py
```

On first run this will:
1. Download the trained classifier checkpoint (`best_model.pt`) from the
   project's Hugging Face model repo, since it isn't found locally.
2. Prompt you for **Issue**, **Topic**, **Headline**, and **News Body**.
3. Print the predicted bias with a full confidence breakdown.

You can also point it at a checkpoint you already have locally:
```bash
python predict.py --checkpoint /path/to/best_model.pt
```

#### Worked example

**Input:**
- Issue: `Healthcare`
- Topic: `Drug Pricing`
- Headline: `Lawmakers Propose Cap on Skyrocketing Prescription Drug Prices to Protect Working Families`
- News Body:
  > In a major push to make healthcare more accessible, a new legislative
  > framework would impose strict caps on the skyrocketing cost of
  > life-saving medications. Consumer advocates have long condemned Big
  > Pharma for rampant price gouging that forces millions of working
  > families to ration their prescriptions. Sponsors of the bill maintain
  > that healthcare should be a fundamental human right, not a
  > profit-driven enterprise designed to enrich corporate shareholders.

**Output:**
```
---------------Prediction---------------------
Predicted Bias : Liberal(Left)
Class Probabilities:
Liberal(Left) : 63.36%  <-----predicted
Center: 29.86%
Conservative: 6.78%
--------------------------------------------------------------
```

#### How to read the output

- **Predicted Bias** — the single class the model considers most likely
  (highest probability).
- **Class Probabilities** — a softmax distribution over all three classes
  (sums to ~100%), i.e. the model's confidence, not just a binary decision:
  - A **lopsided** distribution (like 63% / 30% / 7% above) means the model
    found strong framing cues pointing one direction.
  - A **close** distribution (e.g. 40% / 35% / 25%) means the article's
    framing was more ambiguous or mixed — treat the top prediction with
    lower confidence in that case.


## Tech stack

Python · PyTorch · Hugging Face Transformers & Hub (RoBERTa) · scikit-learn · pandas / NumPy

## License

[MIT](LICENSE)