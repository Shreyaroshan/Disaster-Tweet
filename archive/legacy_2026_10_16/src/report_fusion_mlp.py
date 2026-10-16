"""Final-submission comparison, including unsuccessful experiments."""
import argparse
import json
import numpy as np
import pandas as pd
from project import ROOT
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-name',default='mlp_fusion_2026_10_16');args=parser.parse_args()
    out=ROOT/'experiments'/args.run_name
    summary=json.loads((out/'summary.json').read_text());selected=json.loads((out/'selection.json').read_text());cv=json.loads((out/'cv_selection.json').read_text())
    earlier=ROOT/'experiments/mlp_final_2026_10_16'
    prior=json.loads((earlier/'summary.json').read_text())['comparison'][1]
    prior_cv=json.loads((earlier/'cv_selection.json').read_text())
    reference=next(r for r in prior_cv['ranking'] if r['name']=='candidate_01')
    old,new=summary['comparison'];c=selected['config']
    rows=[dict(model='September MLP',**{k:old[k] for k in ['accuracy','precision','recall','f1']}),
          dict(model='Embedding-only attempt',**{k:prior[k] for k in ['accuracy','precision','recall','f1']}),
          dict(model='Lexical + semantic MLP',**{k:new[k] for k in ['accuracy','precision','recall','f1']})]
    pd.DataFrame(rows).to_csv(out/'all_attempts_comparison.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,5.3));x=np.arange(4)
    for shift,row,color in zip([-.26,0,.26],rows,['#94a3b8','#d99a39','#087e8b']):
        bars=ax.bar(x+shift,[100*row[k] for k in ['accuracy','precision','recall','f1']],width=.24,label=row['model'],color=color)
        ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
    ax.set_xticks(x,['Accuracy','Precision','Recall','F1']);ax.set_ylim(0,100);ax.set_ylabel('Percent');ax.set_title('MLP final-submission experiments: existing local test set')
    ax.legend(loc='upper center',ncol=3,frameon=False,fontsize=9);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
    predictions=pd.read_csv(out/'test_predictions.csv');cm=confusion_matrix(predictions.target,predictions.prediction,labels=[0,1])
    fig,ax=plt.subplots(figsize=(5,4.5));ax.imshow(cm,cmap='Blues')
    for (i,j),value in np.ndenumerate(cm): ax.text(j,i,str(value),ha='center',va='center',fontsize=18,color='white' if value>cm.max()/2 else 'black')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Non-disaster','Disaster'],yticklabels=['Non-disaster','Disaster'],xlabel='Predicted',ylabel='Actual',title='Lexical-semantic MLP: test confusion matrix')
    fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=180);plt.close(fig)
    rank=cv['ranking'][0]
    lines=['# MLP final-submission experiments — October 16, 2026','',
           f"The selected lexical-semantic MLP scores **{new['accuracy']*100:.2f}% accuracy** and **{new['f1']*100:.2f}% disaster-class F1** on the existing 1,023-tweet local test split.",'',
           '## Complete comparison — including the unsuccessful attempt','',
           '| Model | Accuracy | Precision | Recall | F1 |','|---|---:|---:|---:|---:|']
    for row in rows: lines.append('| '+row['model']+' | '+' | '.join(f"{row[k]*100:.2f}%" for k in ['accuracy','precision','recall','f1'])+' |')
    lines+=['',f"Relative to September: accuracy {(new['accuracy']-old['accuracy'])*100:+.2f} percentage points; F1 {(new['f1']-old['f1'])*100:+.2f} points. Correct test predictions: {cm.trace()}/1,023 (September: 836). Detected disasters: {cm[1,1]}/416; missed disasters: {cm[1,0]}; false alarms: {cm[0,1]}/607 non-disasters.",'',
            '![Comparison](comparison.png)','',
            'The first October attempt combined Wikipedia and Twitter embeddings. It improved the validation F1 but reduced test accuracy to 80.55% and F1 to 76.11%. That result was inspected before the follow-up was designed and is retained here. The follow-up therefore remains adaptive and exploratory; its test result is not a fresh independent confirmation.','',
            '## What the selected model does','',
            f"1. Semantic input: {'frozen' if c['frozen'] else 'fine-tuned'} {c['sources']} word embeddings, masked mean and max pooling, LayerNorm, dropout 0.3, Dense({c['hidden']}) and ReLU.",
            f"2. Lexical input: training-fitted word unigram/bigram TF-IDF (up to 8,000 features){' plus character 3–5-gram TF-IDF (up to 8,000 features, weight 0.5)' if c['characters'] else ''}; feature scale 5, dropout 0.3, Dense({c['hidden']}) and ReLU.",
            f"3. Concatenate both hidden representations; dropout {c['dropout']}, Dense({c['hidden']}), ReLU, dropout, then one output logit.",'',
            'This remains a feed-forward MLP classification method with two text representations. It does not contain a CNN, LSTM or Transformer. It changes the input representation and classifier relative to the original embedding-average MLP, so describe it explicitly in the methodology rather than calling it only a learning-rate adjustment.','',
            f"Selected configuration: {selected['config']['name']}. Deployment uses {'an equal-weight three-seed MLP probability ensemble' if selected['ensemble'] else 'one MLP with predesignated seed 42'}, seeds {selected['seeds']}; threshold {selected['threshold']:.2f}. Training uses AdamW, learning rate 0.001 for dense layers, embedding learning rate 0.0002 only when unfrozen, weight decay 0.01, batch size 64, unweighted BCE, gradient clipping 5, maximum30 epochs and patience6. Validation accuracy selects checkpoints, with F1 breaking ties.",'',
            '## Cross-validation and model selection','',
            'The original 4,770-row training split is divided into three stratified outer assessment folds. Each development fold has a separate stratified 15% inner early-stopping split. All embedding vocabularies, UNK vectors, TF-IDF vocabularies and IDF statistics are fitted on the inner training data only. Both experiments reuse identical fold IDs and seeds42/123. The fusion search compares four configurations over all six fold/seed combinations (24 fits), at fixed threshold0.50.','',
            '| Model family under the same folds | Mean CV accuracy | Descriptive SD | Mean CV F1 |','|---|---:|---:|---:|',
            f"| Current-architecture reference | {reference['accuracy']*100:.2f}% | {reference['accuracy_std']*100:.2f} points | {reference['f1']*100:.2f}% |",
            f"| Selected fusion configuration | {rank['accuracy']*100:.2f}% | {rank['accuracy_std']*100:.2f} points | {rank['f1']*100:.2f}% |",'',
            f"CV difference: {(rank['accuracy']-reference['accuracy'])*100:+.2f} accuracy points and {(rank['f1']-reference['f1'])*100:+.2f} F1 points. The reference allowed 40 epochs, while the fusion candidates allowed30, both with patience6. Scores are used for selection; this is not an unbiased nested-CV estimate of the whole tuning process. Fold/seed results are correlated, so SD is descriptive rather than a confidence interval. CV fitting sets are smaller than the final fitting set, so do not compare CV accuracy directly with the September test accuracy.",'',
            'The winning configuration was trained on the full original training split at three seeds, using the original validation split for early stopping. Validation compared designated seed42 versus the three-seed ensemble and thresholds0.35–0.65. The final model and threshold were locked before this follow-up’s test evaluation. The test result did not change that choice.','',
            f"Final validation accuracy/F1: {selected['validation']['accuracy']*100:.2f}% / {selected['validation']['f1']*100:.2f}%. Previous validation accuracy/F1: {selected['previous_validation']['accuracy']*100:.2f}% / {selected['previous_validation']['f1']*100:.2f}%. Validation gate requiring both metrics to match or exceed the current model: **{'passed' if selected['validation_gate_passed'] else 'not passed'}**.",'',
            'Total October fitting runs: 33 embedding-only CV fits +5 confirmation fits, followed by24 fusion CV fits +3 confirmation fits =65 fits. Original checkpoints and data remain intact. Each attempt has its own saved plan, fold IDs, histories, checkpoint hashes and predictions.','',
            '## Evidence limits for the final submission','',
            'The original validation and test sets have been reused across several project iterations. These results support an exploratory comparison, not guaranteed performance on future tweets. A new independently labeled test set or nested cross-validation of the entire selection process would strengthen the final claim. All stages, including the failed attempt, should be reported.','']
    for k,interval in summary['paired_bootstrap_95_percent_delta_intervals'].items(): lines.append(f"- {k.upper() if k=='f1' else k.title()} improvement versus September: paired tweet-bootstrap95% interval {interval[0]*100:+.2f} to {interval[1]*100:+.2f} percentage points.")
    lines+=['','These intervals condition on the fitted models and do not capture selection/seed uncertainty or correlations between tweets about the same event. If an interval includes zero, do not describe that gain as statistically established.','',
            'Data: the original Kaggle disaster-tweet training file, with the same existing deduplication and conflicting-label exclusions. No labels were manually changed in October. Pretrained vectors: [Stanford GloVe](https://nlp.stanford.edu/projects/glove/) via the [Gensim mirror](https://github.com/piskvorky/gensim-data), [PDDL license](https://opendatacommons.org/licenses/pddl/1-0/). Source and matrix checksums are preserved in the embedding provenance files. Twitter-corpus overlap with benchmark tweets was not audited; the final fusion candidate’s source choice is listed above.','',
            '## Run the saved model','','```bash',
            './.venv/bin/python src/fusion_mlp.py predict --text "Flooding has forced residents to evacuate"','```','',
            'To reproduce with a new output folder after preparing the same split and wiki100/wiki300 embeddings:','','```bash',
            './.venv/bin/python src/fusion_mlp.py search --run-name mlp_fusion_repeat',
            './.venv/bin/python src/fusion_mlp.py confirm --run-name mlp_fusion_repeat',
            './.venv/bin/python src/fusion_mlp.py evaluate --run-name mlp_fusion_repeat',
            './.venv/bin/python src/report_fusion_mlp.py --run-name mlp_fusion_repeat','```','',
            'The saved fold definition at experiments/mlp_final_2026_10_16/folds.json is required. Raw vectors and checkpoints stay local and are excluded from Git. Load only trusted local joblib vectorizers. The unsubmitted kaggle_submission.csv contains predictions for3,263 unlabeled competition tweets; it is not a leaderboard score. AI assisted with code, experiments, tests and reporting; team members should verify and understand the final methodology before submission.','']
    (out/'report.md').write_text('\n'.join(lines))
    print(out/'report.md')


if __name__=='__main__': main()
