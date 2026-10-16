# MLP tuning results — September 17, 2026

The validation-selected MLP improves local test accuracy by 0.78 percentage points and disaster-class F1 by 5.66 points. Original baseline artifacts and reports are preserved.

## Same-test-set comparison

Both models were evaluated on the existing 1,023-tweet labeled test partition (416 disaster tweets). This is not the unlabeled Kaggle competition test set or a leaderboard score.

| Metric | Original MLP | Tuned MLP (selected threshold 0.48) | Change (percentage points) |
|---|---:|---:|---:|
| Accuracy | 75.86% | 76.64% | +0.78 |
| Precision | 77.89% | 71.22% | -6.66 |
| Recall | 56.73% | 71.39% | +14.66 |
| F1 | 65.65% | 71.31% | +5.66 |

The tuned model detects 297 of 416 disaster tweets versus 236 originally: 61 more detected disasters. False positives increase from 67 to 120, explaining the precision decrease. Correct predictions increase from 776 to 784 out of 1,023.

## What changed

- Embedding dimension: 128 → 64; hidden layer remains 64 units.
- Dropout: 0.4 → 0.5.
- Optimizer: Adam → AdamW, with weight decay 0.01; learning rate remains 0.001.
- Loss: class-weighted BCE, using the training negative/positive ratio (2828/1942).
- Maximum epochs: 20 → 40; early-stopping patience: 4 → 6. Selected checkpoint: epoch 11.
- Decision threshold: 0.50 → 0.48, selected using validation F1.
- Training-only vocabulary, preprocessing, batch size 64, and all three data partitions are unchanged.

## Selection and threshold comparison

Twelve predefined configurations were screened with seed 42. The top three were repeated with seeds 123 and 2026. Trial 08 won by mean validation F1 across the three seeds (66.97%, sample standard deviation 3.93 percentage points). Seed 42 was designated for deployment in advance; it was not selected by test performance. Its validation F1 is 71.09% at threshold 0.50 and 71.45% at the selected threshold 0.48. The complete search plan and selection were saved before evaluating the selected model on test labels.

At the conventional 0.50 threshold, the same tuned model obtains 77.13% accuracy and 71.74% F1 on the test set (+1.27 and +6.09 points). The validation-selected 0.48 threshold happens to perform slightly worse on this test set. We retain the validation-selected threshold and report both, rather than changing the selection after seeing test results.

## Limits of the improvement

A 2,000-sample paired tweet bootstrap gives a 95% interval of −1.76 to +3.71 percentage points for accuracy change, and +2.18 to +9.47 points for F1 change. Accuracy improvement is therefore small and uncertain. These intervals condition on the fitted models and do not capture training-seed variation, validation-search uncertainty, or correlation between tweets about the same event. The test partition had already been evaluated for the baseline; a fresh external test set would strengthen any generalization claim. No further tuning used these test outcomes.

## Files and verification

- `src/tune_mlp.py`: reproducible search; choose a new run name to avoid overwriting results.
- `src/predict_tuned_mlp.py`: inference using the saved threshold.
- `mlp_tuned.pt`: selected local checkpoint (excluded from Git).
- `search_plan.json`, `all_runs.json`, `selection.json`: configurations and selection evidence.
- `comparison.csv`, `summary.json`, and prediction CSVs: measured results.
- `comparison.png`: visual comparison.
- Reloaded checkpoint reproduces saved probabilities and labels; all metrics were recomputed; all 18 runs and three-seed rankings were checked; baseline checkpoint, results, and split SHA-256 hashes are unchanged.

Tuning code and this analysis were AI-assisted. Team members should review the experiment and interpret the tradeoff before including it in the submission.
