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
