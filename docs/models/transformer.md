# TRANSFORMER model review card

Assigned reviewer: Member 3. Confirm the member's name in the contribution statement. This implementation and its initial experiments were AI-assisted; review/reproduction/actual changes should be documented before pushing.

## Architecture

Frozen GloVe 100d → projection 64d → sinusoidal positions and normalization → one encoder block with 4 heads and feed-forward 128d/GELU → masked mean+max → Dense(64)/GELU → one logit. Attention is bidirectional; there is no decoder or pretrained BERT.

## Measured selected result

- Local test accuracy: 80.55%.
- Precision: 79.40%; recall: 70.43%; positive-class F1: 74.65%.
- ROC AUC: 0.8670; threshold: 0.47.
- Parameters: 1,257,413; selected deployed models: 1.
- Data: exactly the shared 4,770/1,022/1,023 train/validation/test split.
- These are exploratory reused-local-test results, not Kaggle leaderboard scores or a guarantee on future events.

## Shared behavior

Use disaster_tweets.preprocessing for cleaning/token IDs. Use the shared trainer/evaluator; do not duplicate cleaning or create a different split. BCEWithLogitsLoss consumes logits; sigmoid converts logits to probabilities only for evaluation/inference. Training-only vocabulary and padding masks prevent leakage and spurious padding features. Model class, selected config, methodology card, saved results and artifact manifest are the member's upload scope. The supplied results were generated centrally with AI assistance; document your actual review or reproduction rather than claiming authorship of those earlier runs.

## Run

From the prepared repository with this module copied into its correct path:

```bash
python src/prepare_data.py
python src/train.py --model transformer --run member3_transformer_review
```

The published final checkpoint can be used locally with:

```bash
python src/predict.py --model transformer --text "Flooding has forced residents to evacuate"
```

The final artifacts are local and not included in Git. On a new computer, train to recreate them. For inference on a newly trained run, add `--run member3_transformer_review`.

## Selected configuration

```json
{
  "batch_size": 64,
  "learning_rate": 0.0005,
  "embedding_learning_rate": 0.0002,
  "weight_decay": 0.01,
  "epochs": 30,
  "patience": 6,
  "frozen": true,
  "dropout": 0.3,
  "d_model": 64,
  "heads": 4,
  "layers": 1,
  "feedforward": 128
}
```

## Owner review and viva preparation

1. Explain how this model handles word order and padding.
2. Identify which parameters are pretrained, frozen or fine-tuned.
3. Explain the loss, sigmoid, threshold and validation-only selection.
4. Recompute metrics from the saved predictions and review false positives/negatives.
5. Explain why its accuracy/F1 differ and why the reused-test limitation matters.
6. Review and adapt the corresponding report methodology section in your own words, documenting actual work and commit evidence.

Do not present the initial generated runs or an upload alone as independently authored implementation work. Sign only the contributions you verified.

## Package review — 18 October 2026

Codex-assisted review imported the Member 3 package and inspected the forward
pass, shared preprocessing, BCEWithLogitsLoss, frozen embedding handling and
validation-only selection policy. No model or supplied result files were changed.

Verification performed locally:

- Recomputed accuracy, precision, recall, F1, ROC AUC and confusion counts from
  all 1,023 saved test predictions; all matched the supplied metrics.
- Recomputed validation metrics for all eight candidate runs (1,022 rows each)
  and the selected seed-42 run at threshold 0.47; all matched.
- Confirmed selected configuration and artifact member references agree.
- Checked the 1,257,413 parameter count, right-padding and batch invariance,
  finite empty-input outputs, frozen embedding gradients and finite BCE gradients.
- Ran the existing cleaning and distinct Transformer layer initialization tests;
  both passed. Runtime checks used Python 3.12 and temporary CPU PyTorch
  2.14.1, rather than the original pinned 2.14.0 environment.

These checks audit supplied outputs and model behavior; they do not reproduce
training or regenerate checkpoint predictions. Dataset CSVs, embeddings and
checkpoint weights are absent locally. Full data/integration checks were not
run. Earlier AI-assisted experimental results retain their original provenance;
this review does not establish personal authorship or a member signature.
