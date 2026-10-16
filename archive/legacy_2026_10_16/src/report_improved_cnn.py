"""Generate a comparison report from completed CNN experiments."""
import argparse
import json
import numpy as np
import pandas as pd
from project import ROOT
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default='cnn_improved_2026_09_18')
    args = parser.parse_args()
    out = ROOT/'experiments'/args.run_name
    summary = json.loads((out/'summary.json').read_text())
    selection = json.loads((out/'selection.json').read_text())
    runs = json.loads((out/'all_runs.json').read_text())
    original, improved = summary['comparison'][:2]
    names = ['accuracy', 'precision', 'recall', 'f1']
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(4)
    for shift, row, color in [(-.18, original, '#94a3b8'), (.18, improved, '#087e8b')]:
        bars = ax.bar(x+shift, [row[k]*100 for k in names], width=.34, color=color, label=row['model'])
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
    ax.set_xticks(x, ['Accuracy', 'Precision', 'Recall', 'F1'])
    ax.set_ylim(0, 100)
    ax.set_ylabel('Percent')
    ax.set_title('CNN improvement: same 1,023-tweet local test set')
    ax.legend(loc='upper center', ncol=2, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(out/'comparison.png', dpi=180)
    plt.close(fig)
    predictions = pd.read_csv(out/'test_predictions.csv')
    cm = confusion_matrix(predictions.target, predictions.prediction, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap='Blues')
    for (i,j), count in np.ndenumerate(cm):
        ax.text(j, i, str(count), ha='center', va='center', fontsize=18, color='white' if count>cm.max()/2 else 'black')
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=['Non-disaster','Disaster'], yticklabels=['Non-disaster','Disaster'], xlabel='Predicted', ylabel='Actual', title='Improved CNN confusion matrix')
    fig.tight_layout()
    fig.savefig(out/'confusion_matrix.png', dpi=180)
    plt.close(fig)
    c = selection['config']
    lines = ['# Improved CNN — September 18, 2026', '',
             f"Final local test accuracy: **{improved['accuracy']*100:.2f}%**, disaster-class F1: **{improved['f1']*100:.2f}%**.", '',
             '| Metric | Original CNN | Improved CNN | Change (percentage points) |', '|---|---:|---:|---:|']
    for name in names:
        lines.append(f"| {name.upper() if name=='f1' else name.title()} | {original[name]*100:.2f}% | {improved[name]*100:.2f}% | {(improved[name]-original[name])*100:+.2f} |")
    lines += ['', f"The new predictor gets **{cm.trace()}/1,023 tweets correct**, detects {cm[1,1]}/416 disasters, misses {cm[1,0]} disasters, and falsely flags {cm[0,1]}/607 non-disasters. The original CNN got 777 correct, detected 254 disasters, missed 162 and falsely flagged 84 non-disasters.", '',
              '![Comparison](comparison.png)', '', '## Selected architecture', '',
              f"GloVe 6B 100-dimensional embeddings → embedding-channel dropout 0.1 → parallel 1D convolutions (widths {c['kernels']}, {c['filters']} filters per width) → ReLU → masked {c['pool'].replace('_',' + ')} pooling → dropout {c['dropout']} → one output logit.", '',
              f"Embeddings are {'frozen' if c['frozen'] else 'fine-tuned at learning rate '+str(c['embedding_lr'])}. AdamW trains the convolution/classification layers at learning rate {c['lr']}, weight decay {c['weight_decay']}; unweighted binary cross-entropy, batch size 64, gradient norm clipping at 5, maximum 30 epochs and early-stopping patience 6. Checkpoints use validation accuracy, with F1 breaking ties.", '',
              f"Deployment: {'equal-weight three-seed CNN probability ensemble' if selection['ensemble'] else 'single CNN with predesignated seed 42'}. Selected configuration `{selection['winner']}`; seeds {selection['seeds']}; validation-selected threshold **{selection['threshold']:.2f}**. " + ('The ensemble requires three CNNs at inference and must be described as an ensemble in the project report.' if selection['ensemble'] else ''), '',
              'Padding-only convolution windows are masked. Short texts retain one zero-padded window. Right-padding is trimmed per batch to save computation without discarding tokens; tests verify predictions are invariant to extra padding and different batch lengths.', '',
              '## Selection and validation', '',
              f"{len(runs)} actual training runs: twelve predefined configurations at seed 42, then the top three configurations repeated at seeds 123 and 2026. Search varies frozen/fine-tuned embeddings, convolution widths, filter counts, dropout, pooling and embedding learning rate. The final configuration is chosen by mean seed validation accuracy (then mean F1). For that configuration, designated seed 42 and the three-seed ensemble are compared over a threshold grid 0.30–0.65. Validation accuracy is primary throughout, with F1 breaking ties. `search_plan.json` was saved before training; `selection.json` and checkpoint checksums were saved before the separate evaluation command read test labels.", '',
              '| Seed for selected configuration | Best epoch | Validation accuracy at 0.5 | Validation F1 at 0.5 |', '|---|---:|---:|---:|']
    for row in runs:
        if row['trial'] == selection['winner']:
            lines.append(f"| {row['seed']} | {row['best_epoch']} | {row['validation']['accuracy']*100:.2f}% | {row['validation']['f1']*100:.2f}% |")
    lines += ['', f"Final deployed predictor at the selected threshold has {selection['validation']['accuracy']*100:.2f}% validation accuracy and {selection['validation']['f1']*100:.2f}% F1.", '',
              f"For transparency, the selected predictor at threshold 0.5 scores {summary['comparison'][2]['accuracy']*100:.2f}% test accuracy and {summary['comparison'][2]['f1']*100:.2f}% F1. The validation-selected threshold is retained regardless of test outcomes.", '',
              '## MLP comparison', '']
    if len(summary['comparison']) > 3:
        mlp = summary['comparison'][3]
        lines += [f"The improved MLP scores {mlp['accuracy']*100:.2f}% accuracy and {mlp['f1']*100:.2f}% F1 on the same split. The improved CNN differs by {(improved['accuracy']-mlp['accuracy'])*100:+.2f} accuracy points and {(improved['f1']-mlp['f1'])*100:+.2f} F1 points. This is a descriptive comparison, not a claim of statistically established superiority.", '']
    lines += ['## Reproducibility and limitations', '',
              'The original preprocessing, 4,770/1,022/1,023 train/validation/test split, training-only vocabulary, baseline checkpoints and results are preserved. No tweet labels were changed. External unsupervised GloVe pretraining is reused from the MLP experiment; only training-vocabulary vectors are loaded. No ID, location or keyword feature is used.', '',
              'Sources: [Stanford GloVe](https://nlp.stanford.edu/projects/glove/), [verified Gensim download](https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz), [PDDL license](https://opendatacommons.org/licenses/pddl/1-0/). `embedding_provenance.json` contains download and fitted-matrix checksums. Training token coverage is 89.14%.', '',
              'The local test set has been reused in previous project iterations, so these results are exploratory. The new CNN configuration and threshold were selected only on validation, but repeated validation searches can also overfit. A fresh external labeled test set or nested cross-validation would strengthen generalization claims. These are not Kaggle leaderboard results.', '']
    for name, interval in summary['paired_bootstrap_95_percent_delta_intervals'].items():
        lines.append(f"- {name.upper() if name=='f1' else name.title()} change versus original CNN: paired tweet-bootstrap 95% interval {interval[0]*100:+.2f} to {interval[1]*100:+.2f} percentage points.")
    lines += ['', 'Intervals condition on the fitted models and exclude search/seed variation and event-level tweet correlation.', '',
              '## Commands', '', 'From the project root, after baseline preparation and the embedding preparation documented in README:', '', '```bash',
              './.venv/bin/python src/improve_cnn.py search --run-name cnn_repeat',
              './.venv/bin/python src/improve_cnn.py evaluate --run-name cnn_repeat',
              './.venv/bin/python src/report_improved_cnn.py --run-name cnn_repeat', '```', '',
              'Use the saved selected model:', '', '```bash',
              './.venv/bin/python src/improve_cnn.py predict --text "Flooding has forced residents to evacuate"', '```', '',
              'Checkpoints stay local and are excluded from Git. This folder includes configuration histories, validation predictions, selection evidence, final test predictions, charts, and `kaggle_submission.csv` (3,263 unlabeled competition tweets; not submitted to Kaggle). AI assisted with implementation, training, tests and this report. Team members should review the code and findings before academic submission.', '']
    (out/'report.md').write_text('\n'.join(line.rstrip() for line in lines))
    print(out/'report.md')


if __name__ == '__main__':
    main()
