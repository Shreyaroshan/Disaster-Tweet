"""Build a measured-results report for a completed improved-MLP experiment."""
import argparse
import json
from pathlib import Path
from project import ROOT
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default='mlp_improved_2026_09_18')
    args = parser.parse_args()
    out = ROOT / 'experiments' / args.run_name
    summary = json.loads((out / 'summary.json').read_text())
    selection = json.loads((out / 'selection.json').read_text())
    runs = json.loads((out / 'all_runs.json').read_text())
    provenance = json.loads((out / 'embedding_provenance.json').read_text())
    comparison = summary['comparison'][:3]
    original, previous, improved = comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    frame = pd.DataFrame(comparison)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(4)
    for offset, row, color, label in zip([-.26, 0, .26], comparison, ['#94a3b8', '#d99a39', '#087e8b'], ['Original', 'September 17 tuning', 'Improved']):
        bars = ax.bar(x + offset, [row[m] * 100 for m in metrics], width=.24, color=color, label=label)
        ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=9)
    ax.set_xticks(x, ['Accuracy', 'Precision', 'Recall', 'F1'])
    ax.set_ylim(0, 100)
    ax.set_ylabel('Percent')
    ax.set_title('MLP improvement on the same 1,023-tweet local test set')
    ax.legend(loc='upper center', ncol=3, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / 'comparison.png', dpi=180)
    plt.close(fig)
    predictions = pd.read_csv(out / 'test_predictions.csv')
    cm = confusion_matrix(predictions.target, predictions.prediction, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap='Blues')
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha='center', va='center', fontsize=18, color='white' if value > cm.max() / 2 else 'black')
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Non-disaster', 'Disaster'],
           yticklabels=['Non-disaster', 'Disaster'], xlabel='Predicted label', ylabel='Actual label', title='Improved MLP: test confusion matrix')
    fig.tight_layout()
    fig.savefig(out / 'confusion_matrix.png', dpi=180)
    plt.close(fig)
    prediction_errors = predictions[predictions.target != predictions.prediction].copy()
    prediction_errors['review_notes'] = ''
    prediction_errors.to_csv(out / 'test_errors_for_future_review.csv', index=False)
    c = selection['config']
    model_text = ('GloVe 6B pretrained 100-dimensional word vectors → masked ' + c['pool'].replace('_', '+') +
                  f" pooling → LayerNorm → dropout → Dense({c['hidden']}) → ReLU → dropout → one output logit. " +
                  ('Embeddings are frozen.' if c['frozen'] else 'Embeddings are fine-tuned at learning rate 0.0002.')) if c['family'] == 'glove' else (
                  f"Training-fitted word TF-IDF (unigrams and bigrams){' plus character 3–5-grams' if c['characters'] else ''} → dropout → Dense({c['hidden']}) → ReLU → dropout → one output logit.")
    best_runs = [r for r in runs if r['trial'] == selection['winner']]
    lines = ['# Improved MLP — September 18, 2026', '',
             f"Selected model: **{improved['accuracy']*100:.2f}% accuracy and {improved['f1']*100:.2f}% disaster-class F1** on the existing local labeled test set.", '',
             '## Measured comparison', '',
             '| Metric | Original MLP | September 17 tuning | Improved MLP | Change vs original (points) | Change vs September 17 (points) |',
             '|---|---:|---:|---:|---:|---:|']
    for metric in metrics:
        lines.append(f"| {metric.upper() if metric=='f1' else metric.title()} | {original[metric]*100:.2f}% | {previous[metric]*100:.2f}% | {improved[metric]*100:.2f}% | {(improved[metric]-original[metric])*100:+.2f} | {(improved[metric]-previous[metric])*100:+.2f} |")
    lines += ['', 'All rows use the same 1,023 held-out labeled tweets, including 416 disaster tweets and 607 non-disaster tweets. These are local results, not Kaggle leaderboard scores.', '',
              f"The improved model gets **{int(cm.trace())}/1,023 correct**, detects **{cm[1,1]}/416 disasters**, misses {cm[1,0]}, and falsely flags {cm[0,1]} non-disasters. Original counts: 776 correct, 236 detected disasters, 180 misses, 67 false alarms. September 17 counts: 784 correct, 297 detected disasters, 119 misses, 120 false alarms.", '',
              '![Comparison](comparison.png)', '', '## What changed', '', model_text, '',
              f"Selected configuration: `{selection['winner']}`. AdamW learning rate {c['lr']}, weight decay {c['weight_decay']}, dropout {c['dropout']}, batch size 64; {'training-count-weighted' if c['weighted'] else 'unweighted'} binary cross-entropy. Maximum 40 epochs with patience 7, checkpoint chosen by validation F1 then accuracy. Gradient norm clipped to 5. This remains an MLP classification approach; no CNN, recurrent network or Transformer was added.", '',
              f"Deployment uses {'the equal-weight probability average of three independently trained MLPs' if selection['ensemble'] else 'one MLP, using the predesignated seed 42'}. Seeds: {selection['seeds']}. Decision threshold: **{selection['threshold']:.2f}**, selected using validation data. " + ('The ensemble requires three models at inference and should be identified as an MLP ensemble in the project report.' if selection['ensemble'] else ''), '',
              '## Search protocol', '',
              f"{len(runs)} real training runs: 16 GloVe MLP configurations (frozen/fine-tuned embeddings, mean/mean+max pooling, two hidden/dropout settings, weighted/unweighted loss) and eight TF-IDF MLP configurations (word/word+character features, two hidden sizes, two weight decays), followed by two additional seeds for each of the top three configurations.", '',
              'The initial search screened configurations by validation F1, then compared the top three across seeds 42, 123 and 2026. After seeing nearly tied F1 but different accuracy, an additional validation-only selection stage prioritized the user’s accuracy objective. Its policy was saved in accuracy_refinement_plan.json before computing threshold comparisons. It compares each finalist’s predesignated seed-42 model and three-seed ensemble over thresholds 0.30–0.65, selecting validation accuracy first and F1 as a tie-breaker. This adaptive refinement is explicitly separate from the original F1-first plan, whose selection is preserved in selection_f1_screening.json. Final configuration, checkpoint checksums, seeds and threshold were saved before a separate evaluation command read test labels. No changes were selected using this round’s test outcomes.', '',
              '| Selected configuration seed | Best epoch | Validation accuracy at 0.5 | Validation F1 at 0.5 |', '|---|---:|---:|---:|']
    for r in best_runs:
        lines.append(f"| {r['seed']} | {r['best_epoch']} | {r['validation']['accuracy']*100:.2f}% | {r['validation']['f1']*100:.2f}% |")
    lines += ['', f"Final deployed model’s validation accuracy/F1 at the selected threshold: {selection['validation_selected_threshold']['accuracy']*100:.2f}% / {selection['validation_selected_threshold']['f1']*100:.2f}%.", '',
              f"For transparency, the same final predictor at the standard 0.50 threshold scores {summary['comparison'][3]['accuracy']*100:.2f}% test accuracy and {summary['comparison'][3]['f1']*100:.2f}% F1. Deployment retains the validation-selected threshold.", '',
              '## Data and pretrained-vector provenance', '',
              'The dataset, deduplication, excluded conflicting normalized texts, and 4,770/1,022/1,023 train/validation/test split remain unchanged. TF-IDF vocabulary and IDF are fit exclusively on training data. GloVe was pretrained externally without these supervised target labels; the embedding table retains the original training-only project vocabulary. Keyword, location and ID are not input features.', '',
              f"Downloaded verified GloVe 6B 100d vectors from the [Gensim data mirror]({provenance['source']}); original [Stanford GloVe project](https://nlp.stanford.edu/projects/glove/) and [paper](https://nlp.stanford.edu/pubs/glove.pdf). Vectors use the [PDDL license](https://opendatacommons.org/licenses/pddl/1-0/). Matched {provenance['matched_training_words']:,}/{provenance['training_vocabulary_words']:,} training vocabulary words; training token coverage {provenance['training_token_coverage']*100:.2f}%. Download MD5 and SHA-256, fitted matrix checksum and vocabulary checksum are recorded in `embedding_provenance.json`.", '',
              '## Limits and uncertainty', '',
              'The original test set has now been reused across project iterations. Although this search did not use test labels for model selection, these are exploratory comparisons, not independent confirmation on a fresh benchmark. More searches can overfit a validation set. A new external labeled test set or nested cross-validation is needed for a stronger generalization claim.', '']
    for metric, interval in summary['paired_bootstrap_95_percent_delta_intervals'].items():
        lines.append(f"- {metric.upper() if metric=='f1' else metric.title()} change vs original: paired tweet-bootstrap 95% interval **{interval[0]*100:+.2f} to {interval[1]*100:+.2f} percentage points**.")
    lines += ['', 'Bootstrap intervals condition on the fitted models and exclude training/search variation and correlations among tweets about the same event. Validation error inspection also revealed ambiguous/context-dependent examples and apparently questionable labels. No labels were manually corrected or examples removed in this tuning round. The result is the best under the recorded search protocol, not a claim of the maximum achievable accuracy.', '',
              '## Reproduce and use', '', 'From the project root:', '', '```bash',
              './.venv/bin/python src/improve_mlp.py predict --text "Flooding has forced residents to evacuate"', '```', '',
              'To repeat the search, first follow `README.md` to prepare baseline data and embeddings, then choose a new run name:', '', '```bash',
              './.venv/bin/python src/improve_mlp.py search --run-name mlp_improved_repeat',
              './.venv/bin/python src/select_mlp_accuracy.py --run-name mlp_improved_repeat',
              './.venv/bin/python src/improve_mlp.py evaluate --run-name mlp_improved_repeat', '```', '',
              'Checkpoints are local and excluded from Git. TF-IDF serialization, if used, must only be loaded from trusted local files. `kaggle_submission.csv` contains predictions for the 3,263 unlabeled competition test tweets; it has not been submitted to Kaggle.', '',
              'Original checkpoints, the September 17 checkpoint, split files and baseline results are preserved, with SHA-256 checks in `input_hashes.json`. Training configurations, epoch histories, selection evidence and test probabilities are saved alongside this report. AI-assisted code, training and reporting should be reviewed and understood by the team before submission.', '']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(out / 'report.md')


if __name__ == '__main__':
    main()
