# Disaster tweet classification

Your September 25 interim project starts here. Two proposed models, MLP and 1D CNN, are implemented in PyTorch and trained from scratch on the Kaggle Disaster Tweets dataset. Results are measured locally; no Kaggle leaderboard submission has been made.

## For teammates cloning from GitHub

Clone this repository, then set up Python 3.13 and run:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python src/project.py train
./.venv/bin/python -m unittest discover -s tests -v
```

Training downloads the dataset and creates the local split CSVs and model checkpoints, which are excluded from Git. The saved notebook outputs and reports can be viewed before training. Select `.venv/bin/python` in your notebook editor to run its cells. The local Mac setup described below applies to the original project folder.

Repository: https://github.com/Shreyaroshan/Disaster-Tweet

## Start with the completed work

- `reports/interim_report.pdf`: readable interim report with measured results, methodology, literature and remaining work.
- `reports/interim_report.md`: editable report text. The official Part B template was not supplied; transfer content into it before submission.
- `notebooks/Project_Walkthrough.ipynb`: step-by-step guide for inspecting data, models and results.
- `reports/comparison.csv`: common held-out results for both models.
- `reports/training.log`: epoch-by-epoch training evidence.
- `artifacts/mlp.pt` and `artifacts/cnn.pt`: trained checkpoints including tokenizer vocabulary and configuration.
- `reports/contributions.md`: contribution matrix and signature blanks to complete with actual work.

## Run a prediction now

Open Terminal and paste:

```bash
cd '/Users/shreyaroshan/Desktop/DL Project'
./.venv/bin/python src/project.py predict --model cnn --text 'A strong earthquake destroyed several buildings'
./.venv/bin/python src/project.py predict --model mlp --text 'This movie was a disaster'
```

The displayed number is the model's uncalibrated probability estimate. Predictions can be wrong, especially for figurative or ambiguous text.

## Reproduce training

The local environment and dataset are already installed. On this Mac:

```bash
cd '/Users/shreyaroshan/Desktop/DL Project'
./.venv/bin/python src/project.py train
./.venv/bin/python -m unittest discover -s tests -v
./.venv/bin/python src/build_report.py
./.venv/bin/python src/render_report.py
```

Training regenerates the same split using seed 42 and overwrites the model checkpoints and numerical results. Preserve a copy of `artifacts/` and `reports/` first if you want to compare runs. PDF generation was verified using the bundled document runtime in this session. ReportLab is also installed in the project environment for future rendering. Rebuilding the walkthrough resets its saved cell outputs; rerun its cells to refresh them.

For a fresh machine, use Python 3.13 and `python3 -m venv .venv`, then `./.venv/bin/python -m pip install -r requirements.txt`. These are the exact tested package versions for this Mac. For a different Python version/platform, compatible package versions may be needed. The downloader uses a pinned public mirror of the Kaggle CSVs; `data/raw/provenance.json` records source URLs and SHA-256 checksums. Original dataset: https://www.kaggle.com/competitions/nlp-getting-started/data

## What the pipeline does

1. Validate the raw dataset schema, IDs, row counts and class balance.
2. Decode HTML, lowercase, replace URLs/mentions, keep hashtag words and negation, and tokenize English alphanumeric words and apostrophes. No stemming or stopword removal.
3. Exclude normalized texts with conflicting labels and empty text; retain one example per same-label normalized text. Audit counts and excluded rows are saved.
4. Stratify unique tweets approximately 70/15/15 into train, validation and test. Identical normalized text cannot cross partitions.
5. Fit a vocabulary on training text only; encode to 64 tokens with PAD=0 and UNK=1.
6. Train both architectures with Adam, BCE-with-logits, batch 64, learning rate 0.001, seed 42, up to 20 epochs, dropout 0.4 and patience 4. Select the checkpoint by validation positive-class F1.
7. Evaluate each selected checkpoint on the same held-out test partition, at threshold 0.5. Save accuracy, precision, recall, F1, parameter count, CPU training time, confusion matrices and errors.

The original Kaggle `test.csv` has no labels. It is used only to export prediction CSVs, never to compute local test accuracy. The local labeled test partition comes from `train.csv`.

## Understand the two methods

**MLP:** token IDs → trainable 128-dimensional embeddings → average of non-padding embeddings → Dense(64) → ReLU → dropout → one logit. Averaging discards word order. BCE-with-logits trains the classifier; sigmoid is used for inference.

**1D CNN:** token IDs → trainable 128-dimensional embeddings → parallel Conv1d branches with widths 3, 4 and 5, 64 filters each → ReLU → max over valid token windows → concatenate 192 features → dropout → one logit. Short tweets use one padded window; padding-only windows are excluded.

Neither model uses pretrained embeddings, location, keyword or ID as an input feature. Both use the same initialization seed and common protocol, but have different parameter counts. The single run is preliminary and does not establish statistical superiority.

## Before September 25

Read `reports/submission_checklist.md`. Review and understand the generated code and report, record each member's real changes/commits, complete signatures, use the official Part B template, and obtain the required institutional checks. No member work or signatures have been fabricated. LLM assistance is documented in `reports/AI_ASSISTANCE.md`. The LSTM and Transformer remain final-phase work.

## MLP tuning — September 17

The original models and reports are preserved. The new experiment is documented in `experiments/mlp_tuning_2026_09_17/report.md`, with all configurations, training histories, validation selection, and test predictions alongside it.

Twelve configurations were screened; the top three were checked across three seeds (18 runs total). Configuration and threshold were selected using validation F1. The selected MLP achieves **76.64% accuracy and 71.31% F1** on the same 1,023-tweet local test set, versus **75.86% and 65.65%** originally. Recall improves while precision decreases. These are local test results, not Kaggle leaderboard scores.

Predict with the locally saved tuned checkpoint:

```bash
./.venv/bin/python src/predict_tuned_mlp.py --text "Flooding has forced residents to evacuate"
```

To reproduce the search after preparing the baseline data, choose a new output name (existing runs are never overwritten):

```bash
./.venv/bin/python src/tune_mlp.py --run-name mlp_tuning_repeat
```

Checkpoints are excluded from Git. For a new run, pass its `mlp_tuned.pt` path to the prediction script with `--checkpoint`.

## Improved MLP — September 18

The new selected model scores **81.72% accuracy, 80.21% precision, 73.08% recall and 76.48% F1** on the same 1,023-tweet local test partition. This is **+5.08 percentage points in accuracy** versus September 17 tuning, and **+5.87 points** versus the original baseline. The existing test set has been reused across project iterations; treat this as an exploratory comparison until verified on fresh labeled data.

The selected model is a single MLP with fine-tuned **GloVe 6B 100d** embeddings, concatenated masked mean/max pooling, LayerNorm, a 128-unit ReLU hidden layer, dropout 0.5, and one output logit. It uses AdamW, unweighted BCE, embedding learning rate 0.0002 and head learning rate 0.001. Validation selected epoch 31 and threshold 0.49. No Transformer or CNN is used inside this MLP.

Thirty training runs compared 24 GloVe/TF-IDF MLP configurations, then checked the strongest three with three seeds each. The initial F1-focused search was followed by a documented validation-only accuracy refinement to reflect your accuracy objective. The selection and threshold were fixed before the separate final test evaluation. Original data splits, model checkpoints and reports remain unchanged.

Detailed evidence: `experiments/mlp_improved_2026_09_18/report.md`, `comparison.csv`, `all_runs.json`, `search_plan.json`, `accuracy_refinement_plan.json`, and `selection.json`. The folder also contains an unsubmitted `kaggle_submission.csv` for the unlabeled competition test data.

Use the improved model:

```bash
./.venv/bin/python src/improve_mlp.py predict --text "Flooding has forced residents to evacuate"
```

Reproduce on a fresh machine after the original pipeline has prepared the data:

```bash
mkdir -p data/embeddings
curl -fL --retry 2 'https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz' -o data/embeddings/glove-wiki-gigaword-100.gz.part
./.venv/bin/python src/improve_mlp.py prepare-embeddings
./.venv/bin/python src/improve_mlp.py search --run-name mlp_improved_repeat
./.venv/bin/python src/select_mlp_accuracy.py --run-name mlp_improved_repeat
./.venv/bin/python src/improve_mlp.py evaluate --run-name mlp_improved_repeat
./.venv/bin/python src/report_improved_mlp.py --run-name mlp_improved_repeat
./.venv/bin/python src/improve_mlp.py predict --run-name mlp_improved_repeat --text "Flooding has forced residents to evacuate"
./.venv/bin/python -m unittest discover -s tests -v
```

GloVe download size is about 134 MB. Downloaded embeddings, fitted vectorizers, validation probability arrays and model weights are excluded from Git; configuration, provenance and results remain reviewable. Checksums validate the download and fitted training vocabulary. [Stanford GloVe](https://nlp.stanford.edu/projects/glove/) vectors are distributed under the [PDDL license](https://opendatacommons.org/licenses/pddl/1-0/).

## Improved CNN — September 18

The selected CNN achieves **79.96% accuracy, 75.42% precision, 75.24% recall and 75.33% F1** on the same 1,023-tweet local test partition. Relative to the original CNN (75.95% accuracy, 67.37% F1), this is **+4.01 accuracy points and +7.96 F1 points**: 41 more correct predictions, 59 more detected disasters, and 18 additional false alarms. The improved MLP still has higher accuracy (81.72%) and F1 (76.48%); the CNN has higher disaster recall.

The new CNN uses fine-tuned GloVe 6B 100d embeddings, embedding-channel dropout 0.1, 64 filters each at widths 2/3/4, masked max pooling, classifier dropout 0.5, and one output logit. AdamW uses head/convolution learning rate 0.001 and embedding learning rate 0.0002. Validation selected a single seed-42 model at epoch 4 and threshold 0.50. The search completed 18 runs: twelve configurations, then the three strongest repeated across three seeds each. Accuracy was the validation selection objective throughout, with F1 breaking ties. The final test result did not alter the selection.

See `experiments/cnn_improved_2026_09_18/report.md` for the full comparison, uncertainty, search protocol and reproducibility details. The original CNN and MLP artifacts remain unchanged. As with the MLP, these are exploratory results on a reused local test set, not independent confirmation on a fresh dataset or Kaggle leaderboard scores.

Use the saved improved CNN:

```bash
./.venv/bin/python src/improve_cnn.py predict --text "Flooding has forced residents to evacuate"
```

After preparing baseline data and the GloVe embeddings using the commands above, repeat the CNN search with a new run name:

```bash
./.venv/bin/python src/improve_cnn.py search --run-name cnn_repeat
./.venv/bin/python src/improve_cnn.py evaluate --run-name cnn_repeat
./.venv/bin/python src/report_improved_cnn.py --run-name cnn_repeat
./.venv/bin/python src/improve_cnn.py predict --run-name cnn_repeat --text "Flooding has forced residents to evacuate"
```

The dated experiment folder contains training histories, checksums, source/runtime metadata, configuration and threshold selection, prediction CSVs, charts, and an unsubmitted Kaggle prediction file. Model weights remain local and are excluded from Git. CNN code, experiments, tests and reporting were AI-assisted; team members should review the implementation and results before submission.

## Final MLP assessment — October 16

**Retain the September MLP for the submission/demo: 81.72% accuracy and 76.48% F1.** This additional round did not establish a further improvement.

| Attempt | Local test accuracy | F1 |
|---|---:|---:|
| Retained September MLP | 81.72% | 76.48% |
| Combined Wikipedia/Twitter embedding MLP | 80.55% | 76.11% |
| Lexical-semantic MLP ensemble | 81.43% | 75.20% |
| Validation-selected MLP probability blend | 80.74% | 75.47% |

The October work completed 65 training fits: three-fold CV with separate inner early-stopping data, fold-local vocabulary/IDF fitting, repeated seeds, larger/Twitter embeddings, lexical word/character features, and MLP ensembles. A final bounded validation mixture search added no training fits. Some candidates improved CV or validation but did not improve the reused local test result. The reports preserve all unsuccessful outcomes. No existing model or dataset split was overwritten; the CNN was not retrained in October.

Read `experiments/mlp_final_assessment_2026_10_16/report.md` and `recommendation.json` for the final recommendation. The full test output is `verification.log`. The retained prediction command remains:

```bash
./.venv/bin/python src/improve_mlp.py predict --text "Flooding has forced residents to evacuate"
```

### Reproduce the October experiments

The baseline data, the September model/results, and the existing GloVe 100d source are prerequisites. Additional verified embedding downloads:

```bash
curl -fL --retry 2 'https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-300/glove-wiki-gigaword-300.gz' -o data/embeddings/glove-wiki-gigaword-300.gz.part
curl -fL --retry 2 'https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-twitter-100/glove-twitter-100.gz' -o data/embeddings/glove-twitter-100.gz.part
./.venv/bin/python src/final_mlp.py prepare
./.venv/bin/python src/final_mlp.py search --run-name mlp_final_repeat
./.venv/bin/python src/final_mlp.py confirm --run-name mlp_final_repeat
./.venv/bin/python src/final_mlp.py evaluate --run-name mlp_final_repeat
./.venv/bin/python src/report_final_mlp.py --run-name mlp_final_repeat
```

The fusion search reuses the saved fold IDs in `experiments/mlp_final_2026_10_16/folds.json` and fits its lexical features from inner training data only:

```bash
./.venv/bin/python src/fusion_mlp.py search --run-name mlp_fusion_repeat
./.venv/bin/python src/fusion_mlp.py confirm --run-name mlp_fusion_repeat
./.venv/bin/python src/fusion_mlp.py evaluate --run-name mlp_fusion_repeat
./.venv/bin/python src/report_fusion_mlp.py --run-name mlp_fusion_repeat
```

`src/ensemble_mlp.py` reproduces the final validation mixture selection using the three dated, locally saved member predictors. Its `--run-name` selects a new output folder; member paths are fixed in that script. It is an experimental alternative, not the recommended demo model. All vector sources and model artifacts have recorded checksums; weights, fitted vectorizers and raw vectors remain excluded from Git.

**Reporting limitation:** the validation and test splits have been reused across iterations, including adaptive follow-ups after unsuccessful test results. These are exploratory comparisons, not fresh independent evaluation. Keep the stronger September model rather than claim an unobserved accuracy gain. October implementation, experiments, tests and reporting were AI-assisted and should be reviewed by the team.
