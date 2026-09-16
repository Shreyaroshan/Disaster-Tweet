"""Build editable interim report and walkthrough from measured artifacts."""
import json, platform, sys
from pathlib import Path
import pandas as pd
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
r=json.loads((ROOT/'reports/results.json').read_text()); e=json.loads((ROOT/'reports/eda.json').read_text())
a,b=r
rows='\n'.join('| '+ ' | '.join([x['model'].upper(),*[f'{x[k]:.4f}' for k in ['accuracy','precision','recall','f1']],str(x['parameters']),f"{x['training_seconds']:.2f}",str(x['best_epoch'])])+' |' for x in r)
text=f'''# Comparative analysis of deep learning architectures for disaster tweet classification

## Interim report for ICT 4442

Interim deadline: 25 September 2026. Experiment date: 16 September 2026.

Team: Shreya Roshan (230911046), Tanisha Mathur (230953110), Adithya Kini (230911008), Arnab Sinha (230911356). Team number: to be completed by the team.

## Progress and objectives

The project classifies whether a tweet describes a real disaster using tweet text alone. Two of the four proposed architectures have been implemented and evaluated: a feed-forward MLP baseline and a one-dimensional CNN. Both learn embeddings from scratch and use the same data preparation and evaluation protocol. On the held-out partition, CNN F1 is {b['f1']:.4f}, compared with {a['f1']:.4f} for MLP. These are preliminary results from one seed; LSTM and encoder-only Transformer experiments remain for the final phase.

The objectives are to compare distinct neural architecture families under a common protocol, measure predictive performance alongside parameter count and training time, and investigate errors involving ambiguous or figurative disaster language.

## Dataset acquisition and exploratory analysis

The dataset is Kaggle Natural Language Processing with Disaster Tweets. The raw training file contains 7,613 labeled tweets: 4,342 non-disaster and 3,271 disaster tweets. The competition test file contains 3,263 unlabeled tweets, giving 10,876 total rows across the two files. This corrects the rounded 10,000 figure in the synopsis.

Files were acquired from a pinned public GitHub mirror of the competition CSVs after a direct Kaggle download was unavailable. This is not an authenticated Kaggle download. The raw files passed row-count, class-count, required-column, unique-ID and nonempty-text checks. SHA-256 values and the exact pinned URLs are recorded in data/raw/provenance.json. These checks validate expected structure and contents but do not establish byte identity against an independently downloaded official archive.

Only text is a model input. ID is retained for traceability; keyword and location are excluded. Missing keyword and location counts are {e['missing_metadata']['keyword']} and {e['missing_metadata']['location']}, respectively. Since those fields are unused, they are not imputed.

## Preprocessing and partitioning

HTML entities are decoded and text lowercased. URLs become urltoken, user mentions become usertoken, and hashtag words are retained without the hash. The tokenizer retains English letters, digits and word-internal apostrophes. Negation and stop words remain. Stemming, lemmatization and pretrained embeddings are not used. Removing punctuation and non-ASCII characters may lose useful cues; this is a documented baseline limitation.

Normalized inputs with conflicting labels are excluded rather than silently relabeled. This removes {e['excluded_conflicting_or_empty_rows']} conflicting or empty rows, involving {e['conflicting_normalized_texts']} conflicting normalized texts. A further {e['removed_same_label_duplicates']} same-label duplicate rows are collapsed. The resulting {e['retained_rows']} unique examples are stratified approximately 70/15/15 using seed 42. Duplicate detection precedes splitting as a fixed data-curation policy. It uses labels only to identify contradictions; excluded examples are outside the reported evaluation population. This exclusion changes the benchmark population and may remove difficult examples.

| Partition | Rows | Disaster | Non-disaster |
|---|---|---|---|
'''+ '\n'.join(f"| {k.title()} | {v['rows']} | {v['positive']} | {v['rows']-v['positive']} |" for k,v in e['splits'].items())+f'''

No identical normalized text or ID crosses partitions. Near-duplicates and related tweets about the same event can still cross partitions; this is not an event-held-out benchmark. Vocabulary fitting uses training text only and produces {e['vocabulary_size']} entries including PAD and UNK. Inputs are right-padded or truncated to 64 tokens. The maximum observed training length is {e['train_length_percentiles']['100']:.0f} tokens.

![Dataset overview](eda.png)

## Method one MLP baseline

Each token maps to a trainable 128-dimensional embedding. Non-padding embeddings are averaged, producing a single tweet vector. A 64-unit dense layer with ReLU, dropout 0.4 and a final scalar linear layer produce a logit. Sigmoid converts the logit to a disaster probability during inference. This is an embedding-averaging feed-forward baseline inspired by deep averaging networks [7], not a reproduction of the original paper. It discards word order, so reversing the same tokens leaves its representation unchanged.

## Method two one-dimensional CNN

Each token maps to a trainable 128-dimensional embedding. Three parallel one-dimensional convolution branches use widths 3, 4 and 5, with 64 filters per branch. ReLU and global max pooling produce 64 features per branch. Concatenating the branches gives 192 features, followed by dropout 0.4 and one output logit. Padding-only convolution windows are masked before pooling; inputs shorter than a kernel use one padded window. This model can learn local phrase patterns, drawing architectural motivation from [3,6]. It uses randomly initialized embeddings and therefore differs from pretrained-embedding CNNs in the literature.

## Common experimental setup

Both models use Adam with learning rate 0.001, binary cross-entropy with logits, batch size 64 and seed 42. The maximum budget is 20 epochs; training stops after four epochs without a strict validation F1 improvement. The checkpoint with the best positive-class validation F1 is retained. The classification threshold is fixed at 0.5, and no test labels are used for checkpoint selection or threshold tuning. There is no hyperparameter search in this interim run.

Runs use CPU with four PyTorch threads and deterministic algorithms on {platform.machine()}, Python {platform.python_version()}. Exact dependencies are in requirements.txt. Training time covers optimization and per-epoch validation, excluding dataset preparation, plotting, checkpoint export and final test evaluation. Parameter counts include the embedding tables and padding row. Different model capacities and inductive biases remain despite the common settings.

Accuracy is the fraction of correct predictions. Precision is TP/(TP+FP), recall is TP/(TP+FN), and positive-class F1 is 2PR/(P+R). The disaster class, label 1, is the positive class. Undefined precision/recall/F1 are assigned zero by the evaluator.

## Preliminary held-out results

| Model | Accuracy | Precision | Recall | F1 | Parameters | Train seconds | Best epoch |
|---|---|---|---|---|---|---|---|
{rows}

The majority-class predictor would achieve {1-e['splits']['test']['positive']/e['splits']['test']['rows']:.4f} accuracy and zero positive-class F1 on this held-out partition. Both neural models exceed that reference. CNN gains {b['f1']-a['f1']:.4f} absolute F1 over MLP in this single run, with higher recall but lower precision. This pattern is consistent with a possible benefit from phrase features, but it does not establish causality or statistical significance. CNN also costs more training time. Both models miss many actual disaster examples; they are research baselines rather than reliable emergency classifiers.

Learning curves show decreasing training loss without consistent validation improvement, indicating overfitting. Early stopping limits further fitting. Future changes should be chosen on validation data, retaining the held-out test for final evaluation. Repeatedly consulting this interim test result when tuning would weaken its value as an unbiased final evaluation.

![MLP learning curves](mlp_learning.png)

![CNN learning curves](cnn_learning.png)

![MLP confusion matrix](mlp_confusion.png)

![CNN confusion matrix](cnn_confusion.png)

## Error analysis and reproducibility evidence

Full predictions and misclassified examples are exported separately for each model. The files include original tweet IDs and text, labels, probability estimates and predictions. Blank category/notes columns support team review; categories have not been automatically assigned or presented as human annotation. Review false positives and false negatives for figurative language, ambiguity, slang, missing context and annotation inconsistency.

Saved checkpoints include the model configuration and training vocabulary, allowing standalone predictions. Automated checks cover split disjointness, training-only vocabulary, preservation of negation/hashtag words, empty and unknown inputs, padding invariance, checkpoint reload equivalence and recalculation of reported metrics from exported predictions. Check reports/test_results.log for the actual test outcome.

## Literature review

'''+(ROOT/'reports/literature_review.md').read_text().replace('# Literature review\n','',1)+'''

## Remaining work and risk plan

| Date | Work | Evidence or risk response |
|---|---|---|
| 16–19 September | Team reviews preprocessing, MLP and CNN; confirms ownership | Run walkthrough and tests; record actual individual changes and commits |
| 20–22 September | Review papers and inspect false positives/negatives | Verify literature claims and annotate real error examples |
| 23–24 September | Integrate official Part B template and contribution statement | Add team number, real contributions, signatures and institutional checks |
| 25 September | Interim submission | Submit report through course MS Form before 11:59 PM |
| 26 September–10 October | Implement LSTM and encoder-only Transformer | Reuse saved splits and vocabulary; start with modest model sizes |
| 11–15 October | Validation-only tuning and repeated seeds | Address overfitting with regularization; record all configurations and seed variability |
| 16–24 October | Final comparison, error analysis and report | Document limitations and genuine individual contributions |
| 26–29 October | Presentation and viva | Each member explains their own model and experimental decisions |
| 31 October | Final report and code | Complete all four models, references and repository submission |

Compute risk is limited for these two small baselines, but Transformer training may need a smaller model or available accelerator. Pretrained variants must be reported separately to distinguish transfer learning from architecture effects. Dataset ambiguity and event overlap remain risks even after duplicate cleanup. Save the present artifacts before subsequent experiments so changes remain auditable.

## Individual contribution statement

'''+(ROOT/'reports/contributions.md').read_text().replace('# Individual contribution record\n','',1)+'''

## Disclosure of assistance

'''+(ROOT/'reports/AI_ASSISTANCE.md').read_text().replace('# Disclosure of LLM assistance\n','',1)+'''

## Submission preparation

The official Part B template was not included with the supplied synopsis and guidelines. This report provides interim content, but the team must transfer it into that template, verify its accuracy, complete its genuine contribution evidence and signatures, and perform the institutional checks before submission. The course submission has not been made.
'''
(ROOT/'reports/interim_report.md').write_text(text)
nb=nbf.v4.new_notebook()
nb.cells=[nbf.v4.new_markdown_cell('''# Disaster Tweet Classification — start here

Two proposed methods: **MLP** and **1D CNN**. Their trained results are already saved. Run cells in order to inspect them. Select the project's `.venv/bin/python` as the notebook interpreter in your editor; the notebook does not retrain automatically.

Read `README.md` first. The dataset is Kaggle's Disaster Tweets, acquired from a pinned public mirror. The report and provenance explain this choice.'''),nbf.v4.new_code_cell('''from pathlib import Path
import sys, json
ROOT = Path.cwd()
if ROOT.name == 'notebooks': ROOT = ROOT.parent
assert (ROOT / 'src/project.py').exists(), 'Open this notebook from the DL Project folder'
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd
from IPython.display import display, Image, Markdown
from project import clean, predict, MLP, CNN, CONFIG
print(ROOT)'''),nbf.v4.new_markdown_cell('## 1. Inspect raw data and labels\nThe competition test file is unlabeled. Local test metrics use a held-out part of the labeled training file.'),nbf.v4.new_code_cell("raw = pd.read_csv(ROOT/'data/raw/train.csv')\ndisplay(raw.head())\ndisplay(raw.target.value_counts())\ndisplay(Image(filename=str(ROOT/'reports/eda.png')))"),nbf.v4.new_markdown_cell('## 2. Understand preprocessing\nNegation remains; URL and mention identities are replaced; hashtag words are retained. Vocabulary is fitted only on the training partition.'),nbf.v4.new_code_cell("print(clean('NOT a #disaster @friend https://example.com'))\nprint(json.dumps(json.loads((ROOT/'reports/eda.json').read_text()), indent=2))"),nbf.v4.new_markdown_cell('## 3. Inspect architectures\nMLP averages embeddings and loses word order. CNN uses trainable filters spanning 3, 4 and 5 tokens to represent local phrases.'),nbf.v4.new_code_cell("vocab=json.loads((ROOT/'data/processed/vocabulary.json').read_text())\nprint(MLP(len(vocab), CONFIG))\nprint(CNN(len(vocab), CONFIG))"),nbf.v4.new_markdown_cell('## 4. Compare measured results\nThese are preliminary results from one seed, using the same held-out tweets and threshold 0.5.'),nbf.v4.new_code_cell("display(pd.read_csv(ROOT/'reports/comparison.csv'))\nfor model in ['mlp','cnn']:\n    display(Image(filename=str(ROOT/f'reports/{model}_learning.png')))\n    display(Image(filename=str(ROOT/f'reports/{model}_confusion.png')))"),nbf.v4.new_markdown_cell('## 5. Try your own tweets\nChange the examples below. Probabilities are uncalibrated and predictions may be wrong.'),nbf.v4.new_code_cell("for text in ['Flood waters have entered several homes', 'This movie was a disaster']:\n    print(text)\n    for model in ['mlp','cnn']: print(predict(model,text))"),nbf.v4.new_markdown_cell('## 6. Review real errors\nAdd your own category and explanation in the exported error files. Do not change labels merely because the model disagrees.'),nbf.v4.new_code_cell("display(pd.read_csv(ROOT/'reports/cnn_errors.csv').head(10))"),nbf.v4.new_markdown_cell('''## 7. Reproduce training when ready

Run these in Terminal from the project folder. Training overwrites saved artifacts; copy them first to preserve a comparison.

```bash
./.venv/bin/python src/project.py train
./.venv/bin/python -m unittest discover -s tests -v
./.venv/bin/python src/build_report.py
```

The literature review, remaining-model timeline, contribution blanks and disclosure are in `reports/`. Team members must complete real contributions and signatures before submission.''')]
nb.metadata['kernelspec']={'display_name':'Python 3 (project .venv)','language':'python','name':'python3'}
(ROOT/'notebooks').mkdir(exist_ok=True); nbf.write(nb,ROOT/'notebooks/Project_Walkthrough.ipynb')
print('Built interim_report.md and Project_Walkthrough.ipynb')
