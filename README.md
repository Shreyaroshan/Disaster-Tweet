# Disaster Tweet Classification

Four architecture families classify disaster-related tweets using one shared cleaning, vocabulary, split, embedding source, training loop and evaluation format. The active implementation is under `src/disaster_tweets`; historical prototypes and unsuccessful experiments are under `archive/legacy_2026_10_16`.

## Measured final comparison

All scores use the same existing 1,023-tweet labeled local test split. They are exploratory because that test split was reused across project iterations.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| MLP | 81.72% | 80.21% | 73.08% | 76.48% |
| CNN | 79.96% | 75.42% | 75.24% | 75.33% |
| Bidirectional LSTM | 80.94% | 76.25% | 77.16% | 76.70% |
| Encoder-only Transformer | 80.55% | 79.40% | 70.43% | 74.65% |

MLP has the highest observed accuracy; LSTM has the highest observed F1/recall. Their small difference does not establish statistical superiority. Selected Transformer attention layers train on this dataset with frozen GloVe input vectors; this is not a pretrained BERT comparison. Full metrics, confusion counts, training costs and paired uncertainty are in `results/final_2026_10_16`.

## Repository layout

```text
archive/legacy_2026_10_16/       Original source, tests, reports and experiment records
configs/search/                Reproducible search configurations
configs/MODEL.json             Selected config supplied in each member handoff
src/disaster_tweets/
  preprocessing.py             Shared cleaning, split, vocabulary and encoding
  embeddings.py                Verified training-vocabulary GloVe preparation
  models/                      Registry plus member-uploaded model classes
  training.py                  Common fitting and validation selection
  evaluation.py                Common four-model test evaluation
src/prepare_data.py            Data and embedding preparation
src/train.py                   Train one architecture family
src/evaluate.py                Locked four-model evaluation
src/predict.py                 Predict from a selected local checkpoint
results/                       Metrics, histories, predictions and selection evidence
reports/final/                 Final DOCX, Markdown and figures
scripts/                       Figures, report generation and handoff packaging
artifacts/models/              Local selected weights plus tracked manifests
 docs/                         Guidelines checklist, AI disclosure and team workflow
 tests/                        Shared behavior and artifact verification
```

## Team upload plan

The administrator pushes archive/shared infrastructure/results/report. Model implementation files, selected configs and model cards are intentionally left for the assigned members to review and push. Until then, training that model reports an awaiting-upload message; integration tests that require missing modules are skipped. All four implementations were run and verified locally before packaging.

| Assigned reviewer | Model | Upload scope |
|---|---|---|
| Member 1 user | CNN | `src/disaster_tweets/models/cnn.py`, `configs/cnn.json`, `docs/models/cnn.md` |
| Member 2 | MLP | `src/disaster_tweets/models/mlp.py`, `configs/mlp.json`, `docs/models/mlp.md` |
| Member 3 | Encoder-only Transformer | `src/disaster_tweets/models/transformer.py`, `configs/transformer.json`, `docs/models/transformer.md` |
| Member 4 | LSTM | `src/disaster_tweets/models/lstm.py`, `configs/lstm.json`, `docs/models/lstm.md` |

See `docs/TEAM_HANDOFF.md`. The local `handoff` folder contains four ZIP packages. The old three MLP improve/select/report scripts are superseded by this modular structure. Name-to-member mapping needs team confirmation; generated implementation and actual member contributions must be distinguished. No member commits or signatures are fabricated.

## Setup and preparation

Tested with Python 3.13.1, PyTorch 2.14.0, CPU/ARM64, four threads. Use the pinned core requirements; other platforms may need compatible builds.

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python src/prepare_data.py
```

The preparer obtains the 7,613 labeled and 3,263 unlabeled Kaggle-format tweets from a pinned public mirror and verifies row IDs/counts. It preserves the existing deterministic 4,770/1,022/1,023 partitions, excludes conflicting normalized labels, collapses exact normalized duplicates and fits the vocabulary only on training text. IDs, keyword and location are not model inputs. GloVe 6B 100d downloads are about 134 MB and checksum verified. Raw CSVs, vector archives and model weights remain local.

Cleaning decodes HTML, lowercases, replaces URLs/mentions and preserves hashtag words/negation. No stemming or stopword removal is used. All architectures consume the same token IDs and GloVe source. Freezing/fine-tuning and architecture-specific dimensions are selected hyperparameters and disclosed in the report.

## Train and evaluate

After the assigned member has copied/pushed the model module and selected config:

```bash
./.venv/bin/python src/train.py --model mlp --run member2_mlp_review
./.venv/bin/python src/predict.py --model mlp --run member2_mlp_review --text "Flooding has forced residents to evacuate"
```

To reproduce an entire new comparison, choose a new run name and train all four models before evaluation:

```bash
./.venv/bin/python src/train.py --model mlp --run comparison_repeat --config configs/search/mlp.json
./.venv/bin/python src/train.py --model cnn --run comparison_repeat --config configs/search/cnn.json
./.venv/bin/python src/train.py --model lstm --run comparison_repeat --config configs/search/lstm.json
./.venv/bin/python src/train.py --model transformer --run comparison_repeat --config configs/search/transformer.json
./.venv/bin/python src/evaluate.py --run comparison_repeat
```

Selections use validation accuracy with positive-class F1 as tie-breaker. The top two candidates are repeated at seeds 123 and 2026. Validation compares designated seed 42 versus a three-seed ensemble and thresholds 0.35–0.65. Evaluation locks all four selection hashes before reading test labels. Existing output folders are not silently overwritten. Do not use repeated test scores to tune the model.

The local selected final weights can be used with:

```bash
./.venv/bin/python src/predict.py --model cnn --text "Flooding has forced residents to evacuate"
```

Weights are excluded from Git, so a fresh clone must train models locally. All final predictions and unsubmitted Kaggle submission CSVs are provided for audit; they are not Kaggle leaderboard scores.

## Verification and report

```bash
./.venv/bin/python -m unittest discover -s tests -p test_shared_pipeline.py -v
```

Tests cover split/vocabulary leakage, negation/hashtag cleaning, padding/batch invariance, short/empty inputs, frozen gradients, independent Transformer layer initialization and saved predictions/metrics. `results/final_2026_10_16/verification.log` records the completed local run with all four model files available.

The final report is `reports/final/Final_Report_Disaster_Tweet_Classification.docx`; a Markdown version is included for repository reading. Report generation uses Python DOCX/Pillow dependencies from `requirements-report.txt`; the supplied scripts rebuild figures and the document from measured result files. The submitted DOCX was rendered and visually checked. Technical content is complete; the team still needs owner methodology review, confirmed name/role mapping, official Part C template if required, and actual contribution signatures/institutional checks. See `docs/SUBMISSION_CHECKLIST.md`.

## Provenance and assistance

Original dataset: https://www.kaggle.com/competitions/nlp-getting-started/data

Pinned mirror and SHA-256 checksums: `data/raw/provenance.json`; processed split manifest: `data/processed/manifest.json`; vector provenance: `data/embeddings/provenance.json`. GloVe source: https://nlp.stanford.edu/projects/glove/ under PDDL. The integrated literature review cites 13 papers plus the dataset.

AI assistance covered implementation, experiments, tests and report drafting. Each member must review, understand, reproduce/adapt and document their actual work. See `docs/AI_ASSISTANCE.md`. Earlier test reuse, label ambiguity and unequal historical search budgets are disclosed rather than presented as an independent benchmark. The guideline presentations are 26–29 October 2026, with report/code due 31 October 2026.

Local legacy working copies may remain ignored while an existing CNN command finishes. They are archived in Git and are not the active shared implementation.
