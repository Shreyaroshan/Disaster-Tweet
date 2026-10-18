# MLP model review card

Assigned reviewer: Member 2. Confirm the member's name in the contribution statement. This implementation and its initial experiments were AI-assisted; review/reproduction/actual changes should be documented before pushing.

## Architecture

Fine-tuned GloVe 100d → masked mean+max 200d → LayerNorm → dropout 0.5 → Dense(128)/ReLU → dropout → one logit. Pooling discards word order.

## Measured selected result

- Local test accuracy: 81.72%.
- Precision: 80.21%; recall: 73.08%; positive-class F1: 76.48%.
- ROC AUC: 0.8727; threshold: 0.49.
- Parameters: 1,235,157; selected deployed models: 1.
- Data: exactly the shared 4,770/1,022/1,023 train/validation/test split.
- These are exploratory reused-local-test results, not Kaggle leaderboard scores or a guarantee on future events.

## Shared behavior

Use disaster_tweets.preprocessing for cleaning/token IDs. Use the shared trainer/evaluator; do not duplicate cleaning or create a different split. BCEWithLogitsLoss consumes logits; sigmoid converts logits to probabilities only for evaluation/inference. Training-only vocabulary and padding masks prevent leakage and spurious padding features. Model class, selected config, methodology card, saved results and artifact manifest are the member's upload scope. The supplied results were generated centrally with AI assistance; document your actual review or reproduction rather than claiming authorship of those earlier runs.

## Run

From the prepared repository with this module copied into its correct path:

```bash
python src/prepare_data.py
python src/train.py --model mlp --run member2_mlp_review
```

The published final checkpoint can be used locally with:

```bash
python src/predict.py --model mlp --text "Flooding has forced residents to evacuate"
```

The final artifacts are local and not included in Git. On a new computer, train to recreate them. For inference on a newly trained run, add `--run member2_mlp_review`.

## Selected configuration

```json
{
  "batch_size": 64,
  "learning_rate": 0.001,
  "embedding_learning_rate": 0.0002,
  "weight_decay": 0.01,
  "epochs": 40,
  "patience": 7,
  "frozen": false,
  "dropout": 0.5,
  "hidden": 128
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
