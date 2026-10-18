# Disaster Tweet Classification

Four architecture families classify disaster-related tweets using one shared cleaning, vocabulary, split, embedding source, training loop and evaluation format. The active implementation is under `src/disaster_tweets`; historical prototypes and unsuccessful experiments are under `archive/legacy_2026_10_16`.

## Results publication

All four models have been run locally on the same 1,023-tweet test split. These measurements are exploratory because that split was reused across project iterations. The selected encoder-only Transformer uses frozen GloVe input vectors with attention trained on this dataset; it is not pretrained BERT.

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
results/                       Member-uploaded metrics, histories, predictions and selection evidence
reports/final/                 Combined report and figures; uploaded after integration
scripts/                       Figures, report generation and handoff packaging
artifacts/models/              Local weights; manifests uploaded by each member
 docs/                         Guidelines checklist, AI disclosure and team workflow
 tests/                        Shared behavior and artifact verification
```


| Assigned reviewer | Model | Upload scope |
|---|---|---|
| Member 1 user | CNN | `src/disaster_tweets/models/cnn.py`, `configs/cnn.json`, `docs/models/cnn.md`, model-specific results and artifact manifest |
| Member 2 | MLP | `src/disaster_tweets/models/mlp.py`, `configs/mlp.json`, `docs/models/mlp.md`, model-specific results and artifact manifest |
| Member 3 | Encoder-only Transformer | `src/disaster_tweets/models/transformer.py`, `configs/transformer.json`, `docs/models/transformer.md`, model-specific results and artifact manifest |
| Member 4 | LSTM | `src/disaster_tweets/models/lstm.py`, `configs/lstm.json`, `docs/models/lstm.md`, model-specific results and artifact manifest |


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

After the assigned member has pushed the model module and selected config:

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

Weights are excluded from Git, so a fresh clone must train models locally. Member handoffs include final predictions and unsubmitted Kaggle submission CSVs for audit; they are not Kaggle leaderboard scores.



