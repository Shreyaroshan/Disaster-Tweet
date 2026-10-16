"""Create local per-member model packages; no fabricated member commits."""
import json
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    records={r['model']:r for r in json.loads((ROOT/'results/final_2026_10_16/comparison.json').read_text())}
    descriptions={
        'mlp':'Fine-tuned GloVe100d → masked mean+max200d → LayerNorm → dropout0.5 → Dense128/ReLU → dropout → one logit. Pooling discards word order.',
        'cnn':'Fine-tuned GloVe100d → embedding-channel dropout0.1 → Conv1d widths2/3/4 with64 filters each → ReLU → masked max → dropout0.5 → one logit.',
        'lstm':'Fine-tuned GloVe100d → input dropout0.15 → packed bidirectional LSTM64 per direction → masked mean+max256d → LayerNorm → dropout0.4 → Dense64/ReLU → dropout → one logit.',
        'transformer':'Frozen GloVe100d → projection64d → sinusoidal positions and normalization → one encoder block with4 heads and feed-forward128d/GELU → masked mean+max → Dense64/GELU → one logit. Attention is bidirectional; there is no decoder or pretrained BERT.'}
    roles={'cnn':1,'mlp':2,'transformer':3,'lstm':4}
    (ROOT/'docs/models').mkdir(parents=True,exist_ok=True);(ROOT/'handoff').mkdir(exist_ok=True)
    manifests=[]
    for name,number in roles.items():
        selection=json.loads((ROOT/f'results/final_2026_10_16/{name}/selection.json').read_text());config=selection['config'];r=records[name]
        card=f'''# {name.upper()} model review card

Assigned reviewer: Member {number}. Confirm the member's name in the contribution statement. This implementation and its initial experiments were AI-assisted; review/reproduction/actual changes should be documented before pushing.

## Architecture

{descriptions[name]}

## Measured selected result

- Local test accuracy: {r['accuracy']*100:.2f}%.
- Precision: {r['precision']*100:.2f}%; recall: {r['recall']*100:.2f}%; positive-class F1: {r['f1']*100:.2f}%.
- ROC AUC: {r['roc_auc']:.4f}; threshold: {r['threshold']:.2f}.
- Parameters: {r['parameters_per_model']:,}; selected deployed models: {r['deployed_models']}.
- Data: exactly the shared 4,770/1,022/1,023 train/validation/test split.
- These are exploratory reused-local-test results, not Kaggle leaderboard scores or a guarantee on future events.

## Shared behavior

Use disaster_tweets.preprocessing for cleaning/token IDs. Use the shared trainer/evaluator; do not duplicate cleaning or create a different split. BCEWithLogitsLoss consumes logits; sigmoid converts logits to probabilities only for evaluation/inference. Training-only vocabulary and padding masks prevent leakage and spurious padding features. Model class, selected config and this methodology card are the member's upload scope.

## Run

From the prepared repository with this module copied into its correct path:

```bash
python src/prepare_data.py
python src/train.py --model {name} --run member{number}_{name}_review
```

The published final checkpoint can be used locally with:

```bash
python src/predict.py --model {name} --text "Flooding has forced residents to evacuate"
```

The final artifacts are local and not included in Git. On a new computer, train to recreate them. For inference on a newly trained run, add `--run member{number}_{name}_review`.

## Selected configuration

```json
{json.dumps(config,indent=2)}
```

## Owner review and viva preparation

1. Explain how this model handles word order and padding.
2. Identify which parameters are pretrained, frozen or fine-tuned.
3. Explain the loss, sigmoid, threshold and validation-only selection.
4. Recompute metrics from the saved predictions and review false positives/negatives.
5. Explain why its accuracy/F1 differ and why the reused-test limitation matters.
6. Review and adapt the corresponding report methodology section in your own words, documenting actual work and commit evidence.

Do not present the initial generated runs or an upload alone as independently authored implementation work. Sign only the contributions you verified.
'''
        for a,b in [('GloVe100d','GloVe 100d'),('mean+max200d','mean+max 200d'),('dropout0.','dropout 0.'),('Dense128','Dense(128)'),('widths2','widths 2'),('with64','with 64'),('LSTM64','LSTM 64'),('mean+max256d','mean+max 256d'),('Dense64','Dense(64)'),('projection64d','projection 64d'),('with4','with 4'),('feed-forward128d','feed-forward 128d')]:card=card.replace(a,b)
        path=ROOT/f'docs/models/{name}.md';path.write_text(card)
        files=[f'src/disaster_tweets/models/{name}.py',f'configs/{name}.json',f'docs/models/{name}.md']
        instructions=f'''Member {number}: {name.upper()} handoff

Pull the shared repository first. Create your own branch. Copy these files into the identical repository paths, review/run/understand them and document actual changes. Do not restore the old improve/tune scripts into active src.

Files:
'''+''.join('- '+f+'\n' for f in files)+f'''\nAfter review, stage only these files:

git add {' '.join(files)}
git commit -m "Add reviewed {name.upper()} model and methodology"
git push origin YOUR_BRANCH

Use your own Git account and normal review process. The administrator did not create a commit under your identity. Weights/data are excluded. All model files become runnable together after all assigned members upload their modules.
'''
        zip_path=ROOT/f'handoff/member_{number}_{name}.zip'
        with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as bundle:
            for filename in files:bundle.write(ROOT/filename,filename)
            bundle.writestr(f'HANDOFF_member_{number}_{name}.txt',instructions)
        manifests.append({'member':number,'model':name,'zip':str(zip_path.relative_to(ROOT)),'files':files})
    (ROOT/'handoff/manifest.json').write_text(json.dumps(manifests,indent=2))
    print(json.dumps(manifests,indent=2))


if __name__=='__main__':main()
