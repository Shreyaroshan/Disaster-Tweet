"""Final bounded check of all-MLP probability averaging; validation selection only."""
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
from project import ROOT,save_json
from improve_mlp import load_selected,selected_probabilities,metrics,digest
import final_mlp
import fusion_mlp

DEFAULT_RUN='mlp_ensemble_2026_10_16'
MEMBERS=['september','dual_embedding','feature_fusion']
PATHS=[ROOT/'experiments/mlp_improved_2026_09_18',ROOT/'experiments/mlp_final_2026_10_16',ROOT/'experiments/mlp_fusion_2026_10_16']


def member_probabilities(texts):
    return np.array([selected_probabilities(load_selected(PATHS[0]),list(texts)),
                     final_mlp.predict(final_mlp.load(PATHS[1]),list(texts)),
                     fusion_mlp.predict(fusion_mlp.load(PATHS[2]),list(texts))])


def source_hashes():
    return {str(path.relative_to(ROOT)/'selection.json'):digest(path/'selection.json') for path in PATHS}


def select(out):
    out.mkdir(parents=True,exist_ok=False)
    save_json(out/'plan.json',dict(context='Final bounded adaptive ensemble check after individual candidates failed to improve validation/test. Includes the unchanged September model as a control.',
        members=MEMBERS,weights='Nonnegative weights in quarter increments summing to 1 (15 combinations).',
        threshold='0.40..0.65 step .01; validation accuracy, then F1, then higher weight on September model, then proximity to .5.',
        promotion='Validation accuracy and F1 must both match or exceed the September model; otherwise retain September.',
        test='One final exploratory comparison after saving selection; no further experiments based on this test. Reused test and validation cannot provide independent confirmation.'))
    save_json(out/'input_hashes.json',final_mlp.preservation())
    frame=pd.read_csv(ROOT/'data/processed/validation.csv')
    probabilities=member_probabilities(frame.text.tolist());np.save(out/'member_validation_probabilities.npy',probabilities)
    candidates=[]
    for a in range(5):
        for b in range(5-a):
            weights=np.array([a,b,4-a-b])/4
            p=weights@probabilities
            for t in np.round(np.arange(.4,.651,.01),2):
                candidates.append(dict(weights=weights.tolist(),threshold=float(t),**metrics(frame.target,p,t)))
    candidates.sort(key=lambda r:(-r['accuracy'],-r['f1'],-r['weights'][0],abs(r['threshold']-.5)))
    chosen=candidates[0]
    old=json.loads((PATHS[0]/'selection.json').read_text())['validation_selected_threshold']
    chosen.update(members=MEMBERS,previous_validation=old,validation_gate_passed=chosen['accuracy']>=old['accuracy'] and chosen['f1']>=old['f1'],source_selection_sha256=source_hashes())
    pd.DataFrame(candidates).to_csv(out/'validation_search.csv',index=False)
    pd.DataFrame(dict(id=frame.id,target=frame.target,probability=np.array(chosen['weights'])@probabilities)).to_csv(out/'validation_predictions.csv',index=False)
    save_json(out/'selection.json',chosen)
    print(json.dumps(chosen,indent=2),flush=True)


def predict(out,texts):
    chosen=json.loads((out/'selection.json').read_text())
    assert source_hashes()==chosen['source_selection_sha256']
    # Member loaders independently verify checkpoints/vectorizers against checksums.
    return np.array(chosen['weights'])@member_probabilities(texts)


def evaluate(out):
    if (out/'evaluation_started.json').exists(): raise FileExistsError('Already evaluated')
    chosen=json.loads((out/'selection.json').read_text())
    assert final_mlp.preservation()==json.loads((out/'input_hashes.json').read_text())
    save_json(out/'evaluation_started.json',dict(selection_sha256=digest(out/'selection.json'),time_unix=time.time()))
    frame=pd.read_csv(ROOT/'data/processed/test.csv');p=predict(out,frame.text.tolist())
    rows=frame[['id','text','target']].copy();rows['probability']=p;rows['prediction']=(p>=chosen['threshold']).astype(int);rows.to_csv(out/'test_predictions.csv',index=False)
    oldrows=pd.read_csv(PATHS[0]/'test_predictions.csv');assert np.array_equal(oldrows.id,frame.id)
    old=metrics(frame.target,oldrows.probability,.49);new=metrics(frame.target,p,chosen['threshold'])
    comparisons=[dict(model='September MLP',**old),dict(model='Validation-selected MLP blend',**new)]
    rng=np.random.default_rng(20261016);deltas=[];y=frame.target.to_numpy();bp=oldrows.probability.to_numpy()
    for _ in range(2000):
        idx=rng.integers(0,len(y),len(y));a=metrics(y[idx],bp[idx],.49);b=metrics(y[idx],p[idx],chosen['threshold'])
        deltas.append([b[k]-a[k] for k in ['accuracy','f1']])
    intervals={k:np.percentile(np.array(deltas)[:,i],[2.5,97.5]).tolist() for i,k in enumerate(['accuracy','f1'])}
    save_json(out/'summary.json',dict(comparison=comparisons,improvement={k:new[k]-old[k] for k in new},paired_bootstrap_95_percent_delta_intervals=intervals,
        validation_gate_passed=chosen['validation_gate_passed'],previous_models_and_data_preserved=final_mlp.preservation()==json.loads((out/'input_hashes.json').read_text()),
        caveat='Adaptive validation search after earlier test outcomes were inspected. Exploratory local comparison only; selection fixed before this final evaluation. No independent generalization claim.'))
    pd.DataFrame(comparisons).to_csv(out/'comparison.csv',index=False)
    kaggle=pd.read_csv(ROOT/'data/raw/test.csv');pd.DataFrame(dict(id=kaggle.id,target=(predict(out,kaggle.text.tolist())>=chosen['threshold']).astype(int))).to_csv(out/'kaggle_submission.csv',index=False)
    print(pd.DataFrame(comparisons).to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['select','evaluate','predict']);parser.add_argument('--run-name',default=DEFAULT_RUN);parser.add_argument('--text');args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);out=ROOT/'experiments'/args.run_name
    if args.command=='select': select(out)
    elif args.command=='evaluate': evaluate(out)
    else:
        if args.text is None: parser.error('--text required')
        p=float(predict(out,[args.text])[0]);chosen=json.loads((out/'selection.json').read_text());print(json.dumps(dict(disaster_probability=p,threshold=chosen['threshold'],prediction=int(p>=chosen['threshold'])),indent=2))


if __name__=='__main__':main()
