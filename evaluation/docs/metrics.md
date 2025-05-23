# Evaluation Metrics

Translation quality was evaluated using two widely accepted automatic metrics: **BLEU** and **METEOR**. Each captures different aspects of translation performance and helps compare outputs from traditional NMT models and modern LLMs.

---

## BLEU (Bilingual Evaluation Understudy)

BLEU is a precision-based metric that measures n-gram overlap between a model’s translation and a reference translation.

- Best suited for **literal translations** with consistent word choices
- Performs reliably for **high-resource language pairs** with rigid grammar structures

**Scoring:**

- Range: **0.0 to 1.0** (expressed as a decimal or percentage)
- Higher scores indicate closer surface-level matches.

| Score       | Interpretation                          |
|-------------|------------------------------------------|
| **0.40+**   | Excellent (near-human, fluent output)     |
| **0.30–0.40** | Good quality                            |
| **0.20–0.30** | Understandable but flawed               |
| **0.10–0.20** | Low quality or partially incorrect       |
| **< 0.10**  | Very poor translation or off-topic        |

**Limitations:**

- Does not account for synonyms or word order
- Penalizes legitimate paraphrasing, which may bias against LLMs

---

## METEOR (Metric for Evaluation of Translation with Explicit ORdering)

METEOR offers a more nuanced evaluation by considering:

- Stemming (e.g., “run” vs. “running”)
- Synonyms (e.g., “child” vs. “kid”)
- Word reordering

**Scoring:**

- Range: **0.0 to 1.0**
- METEOR scores may be lower than BLEU in literal translations, but often higher for generated outputs due to its handling of synonyms, stems, and flexible word order

| Score       | Interpretation                          |
|-------------|------------------------------------------|
| **0.60+**   | Very strong (fluent and faithful)         |
| **0.50–0.60** | Good quality                            |
| **0.40–0.50** | Mixed (may contain awkward phrasing)    |
| **< 0.40**  | Often disfluent or semantically incorrect |

**Strengths:**

- Captures semantic similarity, useful for LLM outputs
- More aligned with human judgment of fluency and meaning

---

## Notes on Usage

- Both metrics are used for relative comparison across models
- Neither is perfect on its own - human evaluation is often needed for nuanced distinctions
- METEOR tends to favor natural fluency, while BLEU rewards surface-level accuracy