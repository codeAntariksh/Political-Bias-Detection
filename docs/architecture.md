# Architecture: Issue-Topic Guided Cross-Attention Framework

## 1. Problem Statement

Political bias detection aims to determine the ideological leaning (Left /
Center / Right, per the AllSides taxonomy) of a news article. Bias is rarely
carried by facts alone — different outlets covering the same event often
differ in *framing* rather than in the underlying facts. This project treats
bias detection as a framing-identification problem rather than plain topic
classification.

The model conditions its reading of an article's headline and body on an
explicit **issue-topic context**, built from two extra fields in the AllSides
data: `issue` (broad political domain, e.g. Immigration) and `topic`
(specific sub-topic, e.g. Border Security).

## 2. Core Hypothesis

Two articles can describe the same event with very similar facts yet diverge
sharply in ideological presentation. For example, an "Immigration / Border
Security" story might be framed around humanitarian concerns (left),
factual/procedural reporting (center), or national security (right). The
hypothesis is that this divergence is *contextual*: bias emerges from the
interaction between issue-topic context and how the headline/body present it,
not from isolated words or sentences.

## 3. Design Principles

1. Process headline and body separately — headlines and bodies encode bias differently.
2. Explicitly model issue-topic context rather than folding it into a flat feature vector.
3. Use cross-attention (context queries headline/body) instead of concatenation, so the model learns *what to attend to* given the context.
4. Adaptively fuse headline and body signals per article, since some articles reveal bias mostly in the headline and others mostly in the body.

## 4. Pipeline

```
issue, topic, headline, body (raw text)
            |
      DeBERTa-v3-base (token-level hidden states, not CLS pooling)
            |
  issue_tokens [B,16,768]   topic_tokens [B,16,768]   title_tokens [B,32,768]   body_tokens [B,128,768]
            \                    /                          |                        |
             IssueTopicContextEncoder                       |                        |
             (concat -> TransformerEncoder, 2L/4H)           |                        |
                        |                                    |                        |
                  issue_topic_context [B,32,768] ------ CrossAttentionEncoder ---- CrossAttentionEncoder
                        (query)                          (Q=ctx, KV=title)          (Q=ctx, KV=body)
                                                                |                        |
                                                        AttentionPooling            AttentionPooling
                                                                |                        |
                                                      headline_emb [B,768]        body_emb [B,768]
                                                                \____________  ____________/
                                                                             \/
                                                                       GatedFusion
                                                                             |
                                                                      BiasClassifier
                                                                       (768->512->256->3)
                                                                             |
                                                                  Left / Center / Right
```

## 5. Component Notes

- **DeBERTa-v3-base feature extraction** — the top 4 transformer layers are
  unfrozen and fine-tuned with a small learning rate (`5e-6`); everything else
  is frozen. This keeps training cheap while still letting the encoder adapt.
- **Issue-Topic Context Encoder** — concatenates issue and topic token
  sequences and runs them through a lightweight (2-layer, 4-head) Transformer
  encoder so the two fields can attend to each other before being used as a
  query elsewhere.
- **Cross-attention branches** — two independent stacks (4 layers, 8 heads
  each): one queries the headline tokens, one queries the body tokens, both
  using the issue-topic context as the query. This is the main departure from
  a "concatenate everything and classify" baseline (see §9 in the original
  spec below).
- **Attention pooling** — a learned `Linear(768,1)` scores each sequence
  position and produces a softmax-weighted sum, replacing simple mean
  pooling so the model can emphasize the most diagnostic tokens.
- **Gated fusion** — a small MLP gate decides, per article, how much weight
  to give the headline embedding vs. the body embedding before classification.
- **Classifier** — a 3-layer MLP (768 -> 512 -> 256 -> 3) with GELU and
  dropout.

## 6. Why Cross-Attention Instead of Concatenation

A naive baseline concatenates `[issue; topic; headline; body]` and feeds the
result to a classifier, implicitly assuming every field contributes equally
regardless of content. Cross-attention instead lets the model learn *which
parts of the headline and body matter, given this specific issue-topic
context* — a dynamic, per-example weighting rather than a fixed one.

## 7. Training

- Loss: class-weighted cross-entropy (AllSides labels are imbalanced).
- Optimizer: `AdamW` with two parameter groups — DeBERTa's unfrozen layers at
  `5e-6`, and all task-specific modules (context encoder, cross-attention,
  pooling, fusion, classifier) at `2e-4`.
- LR schedule: linear warmup (10% of steps) + linear decay.
- Early stopping on validation macro-F1 (patience = 10 epochs).
- Gradient clipping at norm 1.0.
- Gate-collapse monitoring: if the gated-fusion gate's standard deviation
  drops below 0.05, it has stopped discriminating between headline and body
  and is flagged during training.

## 8. Known Limitations / Next Steps

- Framing signal is currently learned only implicitly through
  cross-attention; there is no explicit triplet/contrastive objective yet.
  See `docs/research_notes.md` for a sketch of a framing-aware,
  triplet-structured contrastive loss extension.
- Evaluation is on an AllSides-derived split; generalization to other bias
  taxonomies or non-US outlets is untested.
