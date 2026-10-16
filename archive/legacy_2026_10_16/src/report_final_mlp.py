"""Build the final-submission MLP experiment report from measured artifacts."""
import argparse
import json
import numpy as np
import pandas as pd
from project import ROOT
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name',default='mlp_final_2026_10_16')
    args=parser.parse_args()
    out=ROOT/'experiments'/args.run_name
    summary=json.loads((out/'summary.json').read_text())
    selected=json.loads((out/'selection.json').read_text())
    cv=json.loads((out/'cv_selection.json').read_text())
    runs=json.loads((out/'runs.json').read_text())
    confirmed=json.loads((out/'confirmation_runs.json').read_text())
    old,new=summary['comparison']
    keys=['accuracy','precision','recall','f1']
    fig,ax=plt.subplots(figsize=(9,5))
    x=np.arange(4)
    for offset,row,color in [(-.18,old,'#94a3b8'),(.18,new,'#087e8b')]:
        bars=ax.bar(x+offset,[row[k]*100 for k in keys],width=.34,label=row['model'],color=color)
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
    ax.set_xticks(x,['Accuracy','Precision','Recall','F1'])
    ax.set_ylim(0,100);ax.set_ylabel('Percent')
    ax.set_title('Final-project MLP: existing 1,023-tweet local test set')
    ax.legend(loc='upper center',ncol=2,frameon=False)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
    pred=pd.read_csv(out/'test_predictions.csv')
    cm=confusion_matrix(pred.target,pred.prediction,labels=[0,1])
    pd.DataFrame(cm,index=['actual_non_disaster','actual_disaster'],columns=['predicted_non_disaster','predicted_disaster']).to_csv(out/'confusion_matrix.csv')
    config=selected['config']
    reference=next(r for r in cv['ranking'] if r['name']=='candidate_01')
    winner=cv['ranking'][0]
    lines=['# Final-project MLP experiments — October 16, 2026','',
           f"The CV-selected predictor obtains **{new['accuracy']*100:.2f}% local test accuracy** and **{new['f1']*100:.2f}% disaster-class F1**. The previous improved MLP obtained {old['accuracy']*100:.2f}% and {old['f1']*100:.2f}%.",'',
           '## Existing test-set comparison','',
           '| Metric | September MLP | CV-selected final MLP | Change (percentage points) |','|---|---:|---:|---:|']
    for k in keys:
        lines.append(f"| {k.upper() if k=='f1' else k.title()} | {old[k]*100:.2f}% | {new[k]*100:.2f}% | {(new[k]-old[k])*100:+.2f} |")
    lines+=['',f"Correct predictions: {cm.trace()}/1,023 (previous: 836). Detected disasters: {cm[1,1]}/416; missed disasters: {cm[1,0]}; false alarms: {cm[0,1]}/607 non-disasters.",'','![Measured comparison](comparison.png)','',
            '## Cross-validation evidence','',
            'The original 4,770-row training split was divided into three stratified outer assessment folds. Each development fold had its own stratified 15% inner early-stopping split. The outer assessment fold was never used for early stopping or fitting vocabulary, IDF, or UNK vectors. The eight candidates were initially run with seed 42; the top two and the current-architecture reference were repeated with seed 123. Selection used mean outer-fold accuracy, with F1 breaking ties. All CV assessment used a fixed threshold of 0.50. The exact row IDs are in folds.json.','',
            '| Confirmed configuration | Mean CV accuracy | Descriptive SD | Mean CV F1 | Runs |','|---|---:|---:|---:|---:|']
    for row in cv['ranking']:
        lines.append(f"| {row['name']} | {row['accuracy']*100:.2f}% | {row['accuracy_std']*100:.2f} points | {row['f1']*100:.2f}% | {row['runs']} |")
    lines+=['',f"The winning configuration differs from the retrained current-architecture reference by {(winner['accuracy']-reference['accuracy'])*100:+.2f} CV accuracy points and {(winner['f1']-reference['f1'])*100:+.2f} F1 points under the same folds and training protocol. CV models see fewer training examples than the final fitted model, so CV scores should not be directly compared to the previous 81.72% test accuracy. Fold/seed measurements are correlated; the descriptive SD is not an independent confidence interval.",'',
            'This uses separate inner early-stopping data, but the outer CV scores are still used to select the configuration. They are therefore selection estimates, not a fully independent nested-CV estimate of the entire tuning procedure.','',
            '## Final architecture and selection','',
            f"Configuration: {config['name']}. Sources: {config['sources']}. Concatenated static word embeddings are fine-tuned; the pooling layer combines {'training-fitted IDF-weighted mean' if config['idf_pool'] else 'masked mean'} and masked max. The classifier is LayerNorm → dropout {config['dropout']} → Dense({config['hidden']}) → ReLU → dropout → one logit. This is an MLP over pooled static embeddings, not a Transformer, CNN or LSTM.",'',
            f"AdamW: embedding learning rate {config['embedding_lr']}, classifier learning rate {config['lr']}, weight decay {config['weight_decay']}; batch size 64; unweighted BCE; gradient clipping at 5; maximum 40 epochs and patience 6. The winning configuration was retrained at five seeds on the original training split, with the original validation split used for early stopping. Validation compared the predesignated seed42 model with the equal-weight five-seed probability ensemble and thresholds 0.35–0.65. The chosen deployment is {'a five-model MLP ensemble' if selected['ensemble'] else 'one MLP'}, seeds {selected['seeds']}, threshold {selected['threshold']:.2f}.",'',
            f"Final validation accuracy/F1: {selected['validation']['accuracy']*100:.2f}% / {selected['validation']['f1']*100:.2f}%; previous validation accuracy/F1: {selected['previous_validation']['accuracy']*100:.2f}% / {selected['previous_validation']['f1']*100:.2f}%.",'',
            f"The prespecified validation gate (both accuracy and F1 must match or exceed the current model) **{'passed' if selected['validation_gate_passed'] else 'did not pass'}**. " + ('The candidate qualifies as a replacement under that validation rule.' if selected['validation_gate_passed'] else 'Retain the previous model as the default; this candidate remains an experimental result.'),'',
            f"Training count: {len(runs)} CV fits plus {len(confirmed)} final-training fits. Each fitted model has {confirmed[0]['parameters']:,} parameters; deployment uses {len(selected['seeds'])} model(s). All configurations, fold IDs, checkpoint hashes, epoch histories, and held-out fold probabilities are saved. The final selection was saved before this round’s test evaluation; the test outcome did not change it.",'',
            '## Data provenance and limitations','',
            'The original deduplicated train/validation/test split and all prior model artifacts are preserved. Pretrained vectors come from [Stanford GloVe](https://nlp.stanford.edu/projects/glove/) via the [Gensim data repository](https://github.com/piskvorky/gensim-data). Wiki100/wiki300 use Wikipedia 2014 + Gigaword 6B; Twitter100 uses the Twitter 27B corpus. The vectors use the [PDDL license](https://opendatacommons.org/licenses/pddl/1-0/). Download MD5/SHA-256 and matrix checksums are recorded. Twitter URL/user tokens map to the project’s urltoken/usertoken placeholders. No benchmark labels are used in that mapping. Overlap between the external Twitter pretraining corpus and benchmark tweets has not been audited.','',
            'The original validation and test splits have already been reused in previous project iterations. These results are exploratory, not independent confirmation on fresh data. A fresh externally labeled evaluation set would be needed for a stronger final generalization claim. Kaggle’s provided competition test has no labels: the exported submission is not a measured Kaggle leaderboard score.','']
    for key,interval in summary['paired_bootstrap_95_percent_delta_intervals'].items():
        lines.append(f"- {key.upper() if key=='f1' else key.title()} improvement versus September model: paired tweet-bootstrap 95% interval {interval[0]*100:+.2f} to {interval[1]*100:+.2f} percentage points.")
    lines+=['','These intervals condition on the fitted models, exclude search/seed uncertainty, and do not account for event correlation. Do not claim statistical superiority when an interval includes zero.','',
            '## Reproduce and predict','','After preparing baseline data and downloading the three vector files documented in README, use a new run name:','','```bash',
            './.venv/bin/python src/final_mlp.py prepare',
            './.venv/bin/python src/final_mlp.py search --run-name mlp_final_repeat',
            './.venv/bin/python src/final_mlp.py confirm --run-name mlp_final_repeat',
            './.venv/bin/python src/final_mlp.py evaluate --run-name mlp_final_repeat',
            './.venv/bin/python src/report_final_mlp.py --run-name mlp_final_repeat','```','',
            'Use the locally saved final candidate:','','```bash',
            './.venv/bin/python src/final_mlp.py predict --text "Flooding has forced residents to evacuate"','```','',
            'The final candidate is stored separately from the September models. Weights and raw vectors are excluded from Git. The Kaggle prediction CSV has not been submitted. Implementation, training, verification and reporting were AI-assisted; team members should review the work and document their actual contributions before academic submission.','']
    (out/'report.md').write_text('\n'.join(lines))
    print(out/'report.md')


if __name__=='__main__': main()
