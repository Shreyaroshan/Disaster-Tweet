"""Fix all four selections using validation only, then export selected artifacts."""
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from disaster_tweets.utils import ROOT,save_json,sha256


def main():
    target=ROOT/'results/final_2026_10_16'
    if (target/'evaluation_plan.json').exists():raise RuntimeError('Test evaluation already started; cannot change selection')
    for model in ['mlp','cnn','lstm','transformer']:
        candidates=[]
        for source in [target/model/'selection.json',ROOT/f'results/replication_2026_10_16/{model}/selection.json',ROOT/f'results/retained/{model}/selection.json']:
            if source.exists():
                value=json.loads(source.read_text());value['selection_source']=str(source.relative_to(ROOT));candidates.append(value)
        if not candidates:raise FileNotFoundError(f'No validation selection for {model}')
        candidates.sort(key=lambda v:(-v['validation']['accuracy'],-v['validation']['f1'],len(v['members'])))
        chosen=candidates[0]
        chosen['selection_candidates']=[{'source':c['selection_source'],'validation_accuracy':c['validation']['accuracy'],'validation_f1':c['validation']['f1']} for c in candidates]
        folder=ROOT/f'artifacts/models/{model}';folder.mkdir(parents=True,exist_ok=True)
        for member in chosen['members']:
            source=ROOT/member['path'];destination=folder/f"seed_{member['seed']}.pt"
            if source.resolve()!=destination.resolve():shutil.copy2(source,destination)
            member['path']=str(destination.relative_to(ROOT));member['sha256']=sha256(destination)
        save_json(target/model/'selection.json',chosen)
        save_json(ROOT/f'configs/{model}.json',chosen['config'])
        save_json(folder/'manifest.json',{'model':model,'members':chosen['members'],'threshold':chosen['threshold'],'validation':chosen['validation']})
        print(model,chosen['selection_source'],chosen['validation'])


if __name__=='__main__':main()
