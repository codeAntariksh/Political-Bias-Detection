# Research Notes: Framing-Aware Extension (Future Work)

The current model learns framing implicitly through cross-attention between
issue-topic context and headline/body tokens. A natural extension explored
conceptually (not yet implemented) is to make the framing signal explicit:

- **Triplet-structured data**: for a given issue-topic pair, sample matched
  articles across the three labels (Left / Center / Right) reporting on the
  same underlying event, forming (anchor, same-frame, different-frame)
  triplets.
- **Contrastive / geometric objective**: add a triplet or contrastive loss
  term on the pooled headline/body embeddings, pulling same-label framings of
  the same event together and pushing different-label framings apart, in
  addition to the existing cross-entropy classification loss.
- **Why it might help**: the current objective only supervises the final
  class label; a contrastive term would directly encourage the embedding
  space to separate *framing style* from *topic content*, which is closer to
  the actual hypothesis in `architecture.md` (§2).

This is left as a documented direction rather than shipped code, since it
requires constructing matched triplets from AllSides (same issue/topic,
different outlet/label) which the current dataset does not provide directly.
