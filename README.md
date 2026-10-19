# Cross-Bank Analysis of Earnings-Call Q&A

**Cambridge Data Science 2026 · Empirical Project · Team 7 · Bank of England RegTech · October 2026**

This project analyses what analysts ask, and how management answers, in the question-and-answer sections of US bank earnings calls. It applies topic modelling (BERTopic), financial sentiment analysis (FinBERT), time-series trend tests and anomaly detection to compare banks. It ends with a retrieval-augmented generation (RAG) system that answers questions about the calls with cited sources.

| | |
|---|---|
| **Banks** | JPMorgan Chase (22 calls), Citigroup (22 calls), Bank of America (2 calls) |
| **Period** | 1Q 2021 – 2Q 2026 |
| **Unit of analysis** | Three-sentence chunks of analyst questions and management answers (6,945 chunks from 1,166 exchanges) |
| **Language** | Python (Google Colab) |

---

## Research questions

1. **Topics.** What subjects are discussed in the Q&A, and which are distinctive for each bank?
2. **Trajectories.** Which topics emerge, fade or recur over time, and do the banks move together?
3. **Tone.** How does the tone of analysts and management differ across topics and banks?
4. **Unusual calls.** Which calls depart from a bank's usual Q&A pattern?
5. **Retrieval.** Can a RAG system return cited, bank-specific evidence for supervisory questions?

---

## Repository structure

```
CamDS2026_EmpProj_Team7/
├── data/
│   └── raw_files/transcripts/<bank>/        # earnings-call transcript PDFs (one folder per bank)
├── notebooks/
│   └── EmProject/
│       ├── <data preparation notebook>.ipynb  # sections 1.1–1.16: from PDFs to analytical units
│       ├── cross_bank_qa_analysis.ipynb      # main analysis: topics, trajectories, tone, themes, unusual calls, RAG
│       ├── hotspot_topics.ipynb              # earlier analysis: topics in new-vocabulary hotspots
│       ├── topic_trajectories.ipynb          # earlier analysis: trajectories of hotspot topics
│       ├── *.parquet / *.csv / *.xlsx        # intermediate data (manifest, pages, documents, turns, units)
│       └── cross_bank_qa/                    # outputs of the main analysis (see below)
└── README.md
```

---

## Pipeline

```
PDF transcripts
   │  1. Manifest and page extraction (PyMuPDF)
   │  2. Documents and conservative text cleaning
   │  3. Q&A boundary detection and speaker parsing (rule-based, tested on every transcript)
   │  4. Analytical units: speaker turns labelled analyst / management / operator
   ▼
transcripts_analytical_units.parquet
   │  5. Q&A exchanges → three-sentence chunks of questions and answers
   │  6. Sentence embeddings (all-MiniLM-L6-v2)
   │  7. BERTopic: UMAP → HDBSCAN → c-TF-IDF (model selection, stability checks)
   │  8. FinBERT tone per chunk
   ▼
Cross-bank analysis: prevalence tests · trajectories · tone · themes · unusual calls
   ▼
RAG: hybrid retrieval (embeddings + BM25 + re-ranking) → Qwen3-8B with citations
```

---

## Main notebook: `cross_bank_qa_analysis.ipynb`

| Section | Content | Methods |
|---|---|---|
| 1–3 | Setup, Q&A corpus, embeddings | Exchange rule, sentence chunking, cached embeddings |
| 4–6 | Topic model | Grid over minimum topic size (NPMI coherence, diversity), outlier reassignment, reproducibility and stability (adjusted Rand index), topic hierarchy |
| 7 | Topics across banks | Chi-square test on common quarters, adjusted residuals (Bonferroni), Cramér's V |
| 8 | Trajectories across banks | Topic share per call, Mann–Kendall trend test, Theil–Sen slope, Benjamini–Hochberg q-values, trajectory classes, Spearman co-movement, Jensen–Shannon agenda distance |
| 9 | Tone across banks | FinBERT net tone; Welch's t-test on call means with 95% CIs and forest plot |
| 10 | Themes | Topics grouped from the hierarchy (Ward linkage), reviewed and adjusted |
| 11 | Unusual calls | Topic novelty and tone change (modified z-scores); Isolation Forest per bank |
| 12–13 | Review samples and saved outputs | Samples for manual validation of topics, FinBERT and flagged calls |
| 14 | RAG system | Dense + BM25 retrieval, reciprocal rank fusion, cross-encoder re-ranking, Qwen3-8B (4-bit) with cited answers, retrieval evaluation |
| 15 | Findings | Results and limitations |

---

## Key findings

These findings are descriptive and based on 22 calls per bank. The full results, with statistics and limitations, are in section 15 of the main notebook.

- **The banks' agendas differ.** Citigroup's Q&A centres on firm-specific restructuring: costs and transformation, divestments, consent orders and strategic targets. JPMorgan Chase's centres on macroeconomic conditions, the balance sheet, credit and capital regulation (χ² = 1030.5, Cramér's V = 0.40).
- **The difference is stable.** The distance between the banks' topic mixes showed no trend over 2021–2026, but changes in each bank's agenda occurred in the same quarters (topic novelty ρ = +0.57).
- **Market topics move together.** Private credit, deposit betas, recession and reserves rose and fell at both banks at the same time; capital targets, buybacks and restructuring followed bank-specific timing.
- **Few significant trends.** The only significant trend after correction was AI and technology at Citigroup; most topics were present throughout the period.
- **Tone differs in management's answers, not in analysts' questions.** Management tone was higher at Citigroup (+0.27) than at JPMorgan Chase (+0.03); analyst tone was the same at both banks (+0.05). The difference held within topics.
- **Unusual calls are mild departures.** No call broke sharply from its bank's history; the most distinctive calls coincided with mid-2022 macro concerns, Citigroup's consent-order quarter (2Q 2024) and private credit (1Q 2026).

---

## How to run

1. Open the notebook in Google Colab and select **Runtime → Change runtime type → T4 GPU**.
2. Mount Google Drive when prompted. Paths are set in cell 1.1 (`REPO_ROOT`, `WORK_DIR`).
3. Run the data-preparation notebook once to create `transcripts_analytical_units.parquet`.
4. Run `cross_bank_qa_analysis.ipynb` with **Runtime → Run all**.
5. After the first run, set the theme names, moves and exclusions at the top of cell 10.1, then re-run sections 10–15.

**Runtime notes**

- *Caching:* embeddings and FinBERT scores are cached on Drive, so later runs are fast.
- *RAG model download:* the RAG section downloads Qwen3-8B (about 16 GB) on first use and needs about 7 GB of free GPU memory.
- *Out-of-memory errors:* if the RAG section runs out of GPU memory, restart the session and run sections 1–10 and 14 in order.

### Main packages

| Purpose | Packages |
|---|---|
| PDF and text | `pymupdf` 1.26.7, `spacy` (`en_core_web_sm`), `wordfreq` |
| Data | `pandas`, `numpy`, `pyarrow`, `openpyxl` |
| Topic model | `bertopic` 0.17.4, `sentence-transformers`, `umap-learn`, `hdbscan`, `scikit-learn` |
| Tone and LLM | `transformers` ≥ 4.51, `torch`, `bitsandbytes` |
| Statistics and plots | `scipy`, `matplotlib` |
| Retrieval | `rank_bm25` |

### Models

| Role | Model |
|---|---|
| Sentence embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Financial tone | `ProsusAI/finbert` |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Answer generation (RAG) | `Qwen/Qwen3-8B` (Apache 2.0, 4-bit; `Qwen3-1.7B` as CPU fallback) |

---

## Outputs (`notebooks/EmProject/cross_bank_qa/`)

| File | Content |
|---|---|
| `qa_chunks_topics_tone.parquet` | Every chunk with bank, quarter, speaker type, topic, theme and FinBERT scores |
| `cross_bank_results.xlsx` | All result tables (one sheet per analysis) |
| `review_samples.xlsx` | Samples for manual validation of topics, FinBERT labels and unusual calls |
| `topic_info.csv`, `topic_trends.csv`, `theme_mapping.csv` | Topic descriptions, trajectories and theme membership |
| `*.png` | Figures: topic and theme shares over time, agenda distance, tone, unusual calls |
| `bertopic_model/` | Fitted topic model (safetensors) |
| `rag/rag_evaluation.xlsx` | Retrieval evaluation, logged answers for review and RAG settings |

---

## Reproducibility

- *Fixed seeds:* every random step uses a fixed seed (42), and UMAP starts from a PCA layout, so repeated runs give identical topics (verified: ARI = 1.00).
- *Caching:* cached embeddings and FinBERT scores are reused only when the model and the texts are unchanged (checked with a fingerprint).
- *Deterministic answers:* RAG generation uses greedy decoding, so the same question and sources always give the same answer.
- *Visible settings:* all analytical choices (thresholds, topic size, theme decisions) are set at the top of the relevant cells and saved with the outputs.

---

## Limitations

- *Topic stability:* topics are moderately stable across random seeds (mean ARI 0.66); fine distinctions between neighbouring topics should be read with caution.
- *Statistical power:* with 22 calls per bank, trend tests detect only strong changes; Bank of America (2 calls) is described but not tested.
- *Independence:* chunk-level tests treat chunks as independent, so their p-values are understated; tone comparisons between banks therefore use call-level means.
- *Tone measurement:* FinBERT measures the tone of wording, not intent or performance.
- *Interpretation:* links between results and external events are hypotheses to be checked against the transcripts, not causal findings.
- *RAG answers:* these can omit or misstate evidence; quotations used in reporting are checked against the source transcripts.

---

## Data

Transcripts are the publicly released earnings-call transcripts of the banks. Check the terms of the transcript source before redistributing the PDFs or derived full-text files.

## Acknowledgements

Developed for the Bank of England RegTech team as part of the Cambridge Data Science 2026 programme (Team 7, October 2026).
