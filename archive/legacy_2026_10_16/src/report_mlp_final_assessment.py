"""Summarize every October attempt and preserve the strongest observed MLP."""
import json
from project import ROOT,save_json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=ROOT/'experiments'
    output=root/'mlp_final_assessment_2026_10_16';output.mkdir(exist_ok=True)
    attempts=[('Current September MLP','mlp_improved_2026_09_18',2),
              ('Combined-embedding MLP','mlp_final_2026_10_16',1),
              ('Lexical-semantic MLP ensemble','mlp_fusion_2026_10_16',1),
              ('Validation-selected MLP blend','mlp_ensemble_2026_10_16',1)]
    rows=[]
    for label,folder,index in attempts:
        row=json.loads((root/folder/'summary.json').read_text())['comparison'][index]
        rows.append(dict(model=label,**{k:row[k] for k in ['accuracy','precision','recall','f1']}))
    pd.DataFrame(rows).to_csv(output/'comparison.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,5.2));x=np.arange(4)
    for shift,key,color in [(-.18,'accuracy','#087e8b'),(.18,'f1','#64748b')]:
        bars=ax.bar(x+shift,[r[key]*100 for r in rows],width=.34,color=color,label=key.upper() if key=='f1' else key.title())
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
    ax.set_xticks(x,['Retained September\nMLP','Combined\nembeddings','Lexical + semantic\nensemble','Validation-selected\nblend'])
    ax.set_ylim(0,100);ax.set_ylabel('Percent');ax.set_title('Final MLP assessment: newer candidates did not beat the existing model')
    ax.legend(loc='upper center',ncol=2,frameon=False);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(output/'comparison.png',dpi=180);plt.close(fig)
    save_json(output/'recommendation.json',dict(status='retain_existing_model',
        reason='All three October candidates had lower observed test accuracy and F1 than the September MLP. No further improvement established.',
        selected_model='experiments/mlp_improved_2026_09_18/selection.json',
        predict_command='./.venv/bin/python src/improve_mlp.py predict --text "Flooding has forced residents to evacuate"',
        measured_results=rows[0],october_training_runs=65,
        caution='Selection and repeated test comparisons are exploratory. Do not claim a fresh independent test or statistical superiority.'))
    lines=['# Final MLP assessment — October 16, 2026','',
           '**Recommendation: retain the existing September MLP. Further improvement was not established in this round.** Its observed local test accuracy remains **81.72%**, with **76.48% F1**. The original pre-tuning MLP scored75.86% accuracy and65.65% F1; that earlier improvement remains valid as a measured local comparison.','',
           '## All observed test outcomes','',
           '| Model | Accuracy | Precision | Recall | F1 |','|---|---:|---:|---:|---:|']
    for row in rows: lines.append('| '+row['model']+' | '+' | '.join(f"{row[k]*100:.2f}%" for k in ['accuracy','precision','recall','f1'])+' |')
    lines+=['','![All attempts](comparison.png)','',
            'The new methods occasionally improved cross-validation or validation scores, but none improved both final test accuracy and F1 over the existing model. No original checkpoint or split was overwritten. The CNN was not trained or changed in this October round.','',
            '## Work completed','',
            '- Downloaded and checksum-verified 300-dimensional Wikipedia/Gigaword GloVe and 100-dimensional Twitter GloVe vectors.',
            '- Tested eight embedding-based MLP configurations: larger vectors, combined sources, IDF-weighted pooling and regularization changes.',
            '- Used three outer assessment folds inside the original training split, with separate inner early-stopping data. Fitted vocabularies, UNK vectors and IDF on inner training only.',
            '- Repeated finalists and the current-architecture reference at a second random seed:33 CV fits plus5 full-training confirmation fits.',
            '- Tested four lexical-semantic MLP configurations using word/character TF-IDF with static embeddings:24 CV fits plus3 confirmation fits.',
            '- Compared a bounded grid of15 probability mixtures and26 thresholds on validation, including the unchanged model as a control. This added no training fits.',
            '- Total:65 real training fits, plus validation-only mixture selection, saved histories, checkpoint checksums, inference commands, prediction files and tests.','',
            '## What can be said in the final project submission','',
            'The original MLP was improved from75.86% to81.72% local accuracy by the September experiment. Further October experiments with larger embeddings, lexical features and ensembles did not produce a stronger measured test result. This is useful negative evidence: greater model complexity did not guarantee better results in this dataset and protocol. Retaining the simpler existing model is justified by the observed comparisons, while acknowledging that the test set has been reused.','',
            'The reference architecture averaged78.22% CV accuracy under the new smaller training folds; the combined-embedding candidate averaged78.87%, and the lexical-semantic candidate79.42%. These were selection scores from correlated fold/seed runs, not independent confirmation. They should not be compared directly with the full-training81.72% test score.','',
            'The existing labeled test partition contains1,023 tweets and has been evaluated repeatedly. The unsuccessful embedding test was inspected before the lexical follow-up was designed, and the final mixture search was also adaptive. All outcomes are therefore exploratory. Do not describe the October work as a fresh independent benchmark, claim an improvement that did not occur, or claim statistically proven superiority. Fresh independently labeled data would be required to strengthen the final generalization claim.','',
            '## Model to use for the submission/demo','',
            '- Model: `experiments/mlp_improved_2026_09_18/trial_15_seed42.pt`.',
            '- Selection: `experiments/mlp_improved_2026_09_18/selection.json`.',
            '- Representation: fine-tuned GloVe6B100d, masked mean+max pooling, LayerNorm, Dense128/ReLU, dropout0.5 and one output logit.',
            '- Decision threshold:0.49.',
            '- Observed local test results:836/1,023 correct,304/416 disasters detected,112 missed disasters and75 false alarms.','',
            '```bash','./.venv/bin/python src/improve_mlp.py predict --text "Flooding has forced residents to evacuate"','```','',
            '## Detailed experiment records','',
            '- `experiments/mlp_final_2026_10_16/report.md`: embedding-only search, CV protocol, source provenance and unsuccessful test outcome.',
            '- `experiments/mlp_fusion_2026_10_16/report.md`: lexical/semantic MLP methodology and all preceding outcomes.',
            '- `experiments/mlp_ensemble_2026_10_16/plan.json`, `selection.json`, `summary.json`: final bounded validation mixture search and test outcome.',
            '- `tests/test_final_mlp.py`, `tests/test_fusion_mlp.py`, `tests/test_ensemble_mlp.py`: fold separation, fitting scope, padding, saved-model predictions and metric verification.',
            '- `verification.log` in this folder: full final test-suite output.','',
            'Sources: [Stanford GloVe](https://nlp.stanford.edu/projects/glove/), [Gensim data mirror](https://github.com/piskvorky/gensim-data), [PDDL license](https://opendatacommons.org/licenses/pddl/1-0/). Source and fitted-matrix checksums are saved. All new code, experiments and reporting were AI-assisted and require team review and understanding before submission. No Kaggle submission was sent; exported competition prediction files contain no measured leaderboard score.','']
    text='\n'.join(lines)
    for a,b in [('scored75','scored 75'),('and65','and 65'),(':33',': 33'),('plus5','plus 5'),(':24',': 24'),('plus3','plus 3'),('of15','of 15'),('and26','and 26'),('Total:65','Total: 65'),('from75','from 75'),('to81','to 81'),('averaged78','averaged 78'),('candidate79','candidate 79'),('full-training81','full-training 81'),('contains1','contains 1'),('GloVe6B100d','GloVe 6B 100d'),('Dense128','Dense(128)'),('dropout0.5','dropout 0.5'),('threshold:0.49','threshold: 0.49'),('results:836','results: 836'),(',304',', 304'),(',112',', 112'),('and75','and 75')]:text=text.replace(a,b)
    (output/'report.md').write_text(text)
    print(output/'report.md')


if __name__=='__main__':main()
