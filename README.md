# NLP-Driven Customer Sentiment Dashboard (v2)

A rebuild of a customer-sentiment dashboard that replaces lexicon-style scoring
and manually defined theme categories with a pretrained transformer classifier,
unsupervised topic modeling, and a real statistical significance layer.

**Read this before the numbers below**: this project was built inside a
network-restricted sandbox. The section [What actually ran, and what
didn't](#what-actually-ran-and-what-didnt) explains exactly which numbers in
this README come from real, executed code in that sandbox, and which parts
need `torch`/`transformers` installed on a normal machine to execute - which
takes about 5 minutes, is exactly one command, and is described there.

---

## 1. Dataset

**Twitter US Airline Sentiment** (Crowdflower, February 2015; the same
dataset later re-hosted on Kaggle). Fetched for real via the Hugging Face
Hub mirror `osanseviero/twitter-airline-sentiment`.

- **Real row count: 14,640** (verified: `data/raw/tweets.csv` after
  `python scripts/fetch_dataset.py`, and `df.shape` shown below)
- **Review text**: `text` - the tweet content (average 101.7 characters /
  17.7 words after cleaning)
- **Rating/label field**: `airline_sentiment` - a real, human/crowd-annotated
  gold label (positive / neutral / negative), not derived by this project
- **Category/segment field (6 groups)**: `airline` - United (3,822),
  US Airways (2,913), American (2,759), Southwest (2,420), Delta (2,222),
  Virgin America (504)
- **Date field**: `tweet_created` - real timestamps, Feb 16-24, 2015

**Why this dataset fits**: it has all four required fields natively (text,
label, a 6-group segment, and real dates), is well above the "low thousands"
volume bar, and - unusually useful for a portfolio piece - already carries a
genuine human-annotated sentiment label. That means the transformer's
predictions can be checked against real ground truth (see
`scripts/classify_with_model.py`, which reports the agreement rate), rather
than the project's own claims being unverifiable.

**A known quirk in the raw data, not a bug in this code**: 501 tweets contain
the literal string "Cancelled Flightled" (e.g. *"...is flight 882 Cancelled
Flightled and what do I do if it is?"*). This is an artifact of the original
2015 Crowdflower dataset itself (a find/replace error when the corpus was
assembled), reproduced verbatim here. It shows up as a topic-model term
because it's real, unmodified text - not because the pipeline is broken.

## 2. Sentiment classification

**Model: `cardiffnlp/twitter-roberta-base-sentiment-latest`**, not
`distilbert-base-uncased-finetuned-sst-2-english`. Reasoning:

1. **Domain match** - it's trained on ~124M tweets and fine-tuned on
   TweetEval. Our data *is* tweets (mentions, hashtags, informal grammar);
   SST-2 was fine-tuned on movie-review sentences, a different register.
2. **Three-class output** - 21.2% of the real dataset is labeled neutral
   (see the distribution below). SST-2 is binary (POSITIVE/NEGATIVE) with no
   neutral class at all, which would force every neutral tweet into a
   polarity it doesn't have. cardiffnlp's model natively outputs
   `{negative, neutral, positive}`, the same label space as our gold data.

Real gold-label distribution (from the actual dataset, computed in this
sandbox):

| Label | Share |
|---|---|
| negative | 62.7% |
| neutral | 21.2% |
| positive | 16.1% |

This module (`src/classify.py`) replaces lexicon/rule-based scoring (VADER,
TextBlob) entirely - it is a `transformers.pipeline("sentiment-analysis", ...)`
call, nothing else. See the execution-status section for why it wasn't run
inside the build sandbox and what stood in for it there.

## 3. Automatic topic/theme extraction

**TF-IDF + NMF** (`src/topic_model.py`), chosen over LDA for three reasons
specific to this data: tweets are very short (LDA's per-document generative
model works better on longer text), NMF is deterministic run-to-run (LDA's
inference is stochastic unless every seed matches exactly), and NMF's
non-negativity constraint tends to produce cleaner, more separable top-term
lists for short text. Full reasoning is in the module docstring.

**This ran for real** against the real tweet text (scikit-learn is
preinstalled in the build sandbox - no network needed). 8 topics, real
output:

| Topic | Prevalence | Top terms |
|---|---|---|
| Delays & logistics | 59.7% | time, plane, delayed, gate, late, bag, waiting |
| Hold times / phone support | 11.8% | help, hold, need, hours, phone, trying |
| Positive acknowledgement | 7.1% | thanks, great, awesome, reply, response |
| Customer service complaints | 7.0% | service, customer, worst, terrible, poor |
| Cancellations | 5.9% | cancelled, tomorrow, dfw, today |
| Gratitude | 3.7% | thank, great, appreciate, response |
| DM / follow-up requests | 3.5% | dm, follow, sent, info, check |
| Misc / slang | 1.3% | fleek, fleet, rt, lol, yall |

("Delays & logistics" dominating at ~60% tracks with the dataset being
62.7% negative overall and airline complaints skewing heavily toward
delays/cancellations - this is a property of the real data, not a modeling
artifact.)

## 4. Segment and trend visualization

`src/viz.py` builds Plotly figures for: sentiment mix by airline (stacked
bar), mean sentiment score over time by airline (line, using the real
`tweet_created` dates), topic prevalence (bar), and topic share by airline
(heatmap). Rendering requires `streamlit`/`plotly` installed - see execution
status below.

## 5. Statistical significance layer

`src/stats_engine.py` - pure `pandas`/`scipy.stats`, zero framework
dependency, independently unit-tested (`tests/test_stats_engine.py`, 12
tests, **bit-for-bit** against raw `scipy.stats` calls - see
[Validation](#validation)). The significance verdict is generated
mechanically from `p < alpha`; nothing in the module can soften or override
that comparison.

**Four real comparisons, run against the real dataset** (`scripts/run_pipeline.py`,
output reproduced verbatim below):

| # | Comparison | Test | Statistic | p-value | Effect size | Verdict |
|---|---|---|---|---|---|---|
| 1 | United vs. US Airways, sentiment score | Welch's t-test | t = 8.039 | 1.07e-15 | Cohen's d = 0.195 | Significant |
| 2 | All 6 airlines, sentiment score | One-way ANOVA | F = 236.26 | 1.97e-243 | eta² = 0.075 | Significant |
| 3 | Positive/negative proportion across 6 airlines | Chi-square | chi2 = 756.73 | 2.65e-161 | Cramer's V = 0.256 | Significant |
| 4 | Top topic ("delays & logistics") membership across 6 airlines | Chi-square | chi2 = 123.52 | 5.64e-25 | Cramer's V = 0.092 | Significant |

Read literally: airline is a statistically real predictor of sentiment score
here (p well under any conventional threshold), but the *practical* size of
that effect is small-to-moderate (eta² = 0.075 means airline explains about
7.5% of the variance in sentiment score; Cohen's d = 0.195 between the two
biggest airlines is a small effect by conventional benchmarks). With n =
14,640, even small real differences reach significance easily - the p-value
answers "is this real," the effect size answers "how much does it matter,"
and this project reports both rather than only the p-value, on purpose.

**Sentiment score used above**: a confidence-weighted score on [-1, 1]
(`positive → +confidence, neutral → 0, negative → -confidence`), computed in
`src/classify.signed_score`. In this run it's computed from the dataset's
own real `airline_sentiment` / `airline_sentiment_confidence` gold columns
(see execution status below for why, and how to recompute it from the
transformer's own output instead).

## 6. Executive summary (optional)

`src/summary.py` makes one templated call to the Claude API to phrase the
top pain-point topic and its cross-segment significance verdict in 1-2
sentences. It only ever receives numbers this project already computed
(never raw review text), and the app falls back to a template-filled
sentence with no API call if `ANTHROPIC_API_KEY` isn't set. Not exercised
in the build sandbox (no key configured there); wire up `.env` from
`.env.example` to use it.

---

## What actually ran, and what didn't

This project was built in a sandboxed environment whose network egress
policy allowed `huggingface.co` and `github.com` but blocked `pypi.org` /
`files.pythonhosted.org` - so `pip install` could not complete for any
package not already preinstalled.

**Ran for real, in the build sandbox, against the real dataset:**
- Dataset download - all 14,640 rows, via direct HTTPS calls (`requests`,
  already installed) to Hugging Face's public `datasets-server` REST API
  (the same JSON API that powers the HF dataset viewer). No `datasets` /
  `pyarrow` package was needed for this. See `scripts/fetch_dataset.py`.
- Data cleaning (`src/data.py`)
- TF-IDF + NMF topic modeling (`src/topic_model.py`) - `scikit-learn` was
  preinstalled
- All four statistical significance tests (`src/stats_engine.py`) -
  `scipy`/`pandas` were preinstalled
- All 12 unit tests for `stats_engine.py`, verified bit-for-bit against raw
  `scipy.stats` calls

**Could not run in the build sandbox** (`torch`, `transformers`, `plotly`,
and `streamlit` are not installed there, and `pip install` to get them
failed - `pypi.org`/`files.pythonhosted.org` returned `403 host_not_allowed`
from that sandbox's egress gateway even after network settings were
opened up mid-build):
- `src/classify.py` - the actual `transformers.pipeline` forward pass never
  executed. **To be clear about what stood in for it**: the stats and topic
  modules above were run against this dataset's own real,
  human-annotated `airline_sentiment` gold labels (a genuine field in the
  source data, not a lexicon score and not invented by this project) rather
  than against transformer output, purely so the significance layer had a
  real sentiment signal to compute on inside this sandbox. This is **not**
  a substitute for the transformer, and the numbers in section 5 should be
  read as "the significance layer works correctly, demonstrated on real
  human-labeled sentiment" rather than "this is what the transformer said."
- `app.py`'s Streamlit UI and every `plotly` chart in `src/viz.py` - written
  against the documented APIs, never launched or rendered.

**To run the rest yourself** (takes about 5 minutes on a normal machine or
in Google Colab):

```bash
pip install -r requirements.txt
python scripts/fetch_dataset.py        # already done here; safe to re-run
python scripts/classify_with_model.py  # runs the REAL transformer, overwrites
                                        # sentiment_score + stats with model output,
                                        # and prints the model/gold agreement rate
streamlit run app.py
```

`scripts/classify_with_model.py` re-runs the exact same topic-modeling and
stats_engine calls shown in section 5, but on the transformer's own
predictions instead of the gold labels, and prints how often the model
agrees with the real human annotations - a genuine accuracy check you can
see with your own eyes, not a number this README asserts.

---

## Validation

Ran in this sandbox, reproducible via the commands shown:

```
$ python scripts/fetch_dataset.py
Dataset reports 14640 total rows. Fetching in pages of 100...
...
Wrote 14640 real rows to data/raw/tweets.csv

$ python -m pytest tests/test_stats_engine.py -v
============================= test session starts ==============================
collected 12 items
tests/test_stats_engine.py::test_two_groups_matches_raw_scipy_welch PASSED
tests/test_stats_engine.py::test_two_groups_matches_raw_scipy_equal_variance PASSED
tests/test_stats_engine.py::test_two_groups_verdict_is_mechanical PASSED
tests/test_stats_engine.py::test_two_groups_effect_size_known_value PASSED
tests/test_stats_engine.py::test_two_groups_drops_nan PASSED
tests/test_stats_engine.py::test_anova_matches_raw_scipy PASSED
tests/test_stats_engine.py::test_anova_verdict_is_mechanical_on_identical_groups PASSED
tests/test_stats_engine.py::test_anova_eta_squared_bounds PASSED
tests/test_stats_engine.py::test_chi_square_matches_raw_scipy PASSED
tests/test_stats_engine.py::test_chi_square_verdict_is_mechanical PASSED
tests/test_stats_engine.py::test_chi_square_cramers_v_perfect_association PASSED
tests/test_stats_engine.py::test_verdict_text_reflects_significance_flag PASSED
============================== 12 passed in 1.03s ==============================

$ python scripts/run_pipeline.py
Loading and cleaning real dataset...
  14640 rows after cleaning
Fitting TF-IDF + NMF topic model on real tweet text...
  Topic 5 (59.7%): time, plane, delayed, gate, late, bag, like, waiting
  ...
[t-test] United vs US Airways sentiment score:
  t=8.0391, p=1.065e-15, d=0.1950
[ANOVA] sentiment score across all 6 airlines:
  F=236.2610, p=1.972e-243, eta^2=0.0747
[Chi-square] positive/negative proportion across airlines:
  chi2=756.7325, p=2.646e-161, Cramer's V=0.2561
[Chi-square] 'time, plane, delayed' topic membership across airlines:
  chi2=123.5180, p=5.64e-25, Cramer's V=0.0919
```

`data/processed/stats_results.json` and `data/processed/topics.json` in this
repo are the actual output files from that run - not hand-written.

---

## Project structure

```
sentiment-dashboard/
├── app.py                      # Streamlit UI (thin - calls src/ modules)
├── requirements.txt
├── .env.example
├── .gitignore
├── src/
│   ├── data.py                 # load/clean the dataset, standardized schema
│   ├── classify.py             # transformer sentiment classification
│   ├── topic_model.py          # TF-IDF + NMF theme extraction
│   ├── stats_engine.py         # significance testing layer (t-test/ANOVA/chi-square)
│   ├── viz.py                  # Plotly chart builders
│   └── summary.py              # optional Claude API executive summary
├── scripts/
│   ├── fetch_dataset.py        # real dataset download (requests only)
│   ├── run_pipeline.py         # what this sandbox actually ran (see above)
│   └── classify_with_model.py  # run once torch/transformers are installed
├── tests/
│   └── test_stats_engine.py    # 12 tests, bit-for-bit vs raw scipy
└── data/
    ├── raw/tweets.csv          # fetched, real, 14,640 rows
    └── processed/              # reviews.csv, topics.json, stats_results.json
```

## Setup

```bash
git clone <this repo>
cd sentiment-dashboard
pip install -r requirements.txt
cp .env.example .env   # optional: add ANTHROPIC_API_KEY for the executive summary

python scripts/fetch_dataset.py       # re-fetch the real dataset (or use the copy already in data/raw/)
python scripts/run_pipeline.py        # topic model + stats on gold labels (fast, no GPU needed)
python scripts/classify_with_model.py # OPTIONAL but recommended: real transformer pass
streamlit run app.py
```

Run the tests with:

```bash
pytest tests/ -v
```
