"""Validation-only MLP tuning. Baseline artifacts and saved data splits stay unchanged."""
import argparse, copy, hashlib, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from project import ROOT, CONFIG, MLP, encode, probabilities, save_json

def scores(y,p,threshold=.5):
    pred=np.asarray(p)>=threshold
    return {k:float(fn(y,pred,**({} if k=='accuracy' else {'zero_division':0}))) for k,fn in [('accuracy',accuracy_score),('precision',precision_score),('recall',recall_score),('f1',f1_score)]}

def train(c,seed,vocab,x,y,out):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model=MLP(len(vocab),c)
    opt=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    weight=float((y['train']==0).sum()/(y['train']==1).sum()) if c['class_weighted'] else 1.
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight))
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x['train'],y['train']),batch_size=c['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(seed))
    best=-1.;stale=0;history=[];start=time.perf_counter()
    for epoch in range(1,c['epochs']+1):
        model.train();total=0.
        for xb,yb in loader:
            opt.zero_grad();loss=loss_fn(model(xb),yb);loss.backward();opt.step();total+=float(loss.detach())*len(yb)
        p=probabilities(model,x['validation']);m=scores(y['validation'].numpy(),p)
        history.append({'epoch':epoch,'train_loss':total/len(y['train']),**{'validation_'+k:v for k,v in m.items()}})
        if m['f1']>best:
            best=m['f1'];state=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
        else:stale+=1
        if stale>=c['patience']:break
    model.load_state_dict(state)
    run=f"{c['trial']}_seed{seed}"
    pd.DataFrame(history).to_csv(out/f'{run}_history.csv',index=False)
    result={'trial':c['trial'],'seed':seed,'validation_f1':best,'best_epoch':best_epoch,'epochs_run':len(history),'training_seconds':time.perf_counter()-start,'config':c}
    torch.save({'model':'mlp','config':c,'seed':seed,'vocabulary':vocab,'state_dict':state},out/f'{run}.pt')
    print(json.dumps({k:v for k,v in result.items() if k!='config'}),flush=True)
    return result

def load_model(path):
    cp=torch.load(path,map_location='cpu',weights_only=True)
    model=MLP(len(cp['vocabulary']),cp['config']);model.load_state_dict(cp['state_dict']);model.eval()
    return cp,model

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-name',default='mlp_tuning_2026_09_17');args=parser.parse_args()
    out=ROOT/'experiments'/args.run_name
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    paths=[ROOT/'data/processed'/f'{k}.csv' for k in ['train','validation','test']]+[ROOT/'artifacts/mlp.pt',ROOT/'reports/results.json']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    save_json(out/'input_hashes.json',hashes)
    parts={k:pd.read_csv(ROOT/f'data/processed/{k}.csv') for k in ['train','validation']}
    vocab=json.loads((ROOT/'data/processed/vocabulary.json').read_text())
    x={k:encode(df.text.tolist(),vocab,64) for k,df in parts.items()};y={k:torch.tensor(df.target.to_numpy(),dtype=torch.float32) for k,df in parts.items()}
    # Fixed search space, written before any experiments. Test is not used for selection.
    settings=[
      (64,64,.4,.001,.01,False),(64,64,.5,.003,.01,False),
      (128,64,.5,.001,.01,False),(128,64,.5,.003,.01,False),
      (128,128,.5,.001,.05,False),(256,64,.5,.001,.01,False),
      (64,32,.3,.003,.05,True),(64,64,.5,.001,.01,True),
      (128,64,.4,.001,.01,True),(128,64,.5,.003,.01,True),
      (128,128,.6,.001,.05,True),(256,64,.5,.001,.01,True),
    ]
    configs=[]
    for i,(emb,hid,drop,lr,wd,weighted) in enumerate(settings):
        configs.append({**CONFIG,'trial':f'trial_{i+1:02d}','embedding_dim':emb,'hidden_dim':hid,'dropout':drop,'learning_rate':lr,'weight_decay':wd,'class_weighted':weighted,'epochs':40,'patience':6})
    save_json(out/'search_plan.json',{'configs':configs,'screen_seed':42,'confirmation_seeds':[123,2026],'selection':'Top 3 by seed-42 validation F1 at threshold 0.5; winner by mean validation F1 across three seeds. Deploy seed 42, not the best seed.','threshold_grid':np.round(np.arange(.2,.701,.01),2).tolist(),'test_policy':'Evaluate only the selected configuration (seed 42) after writing selection.json. Compare 0.5 and validation-selected thresholds; no further tuning based on test results.'})
    runs=[train(c,42,vocab,x,y,out) for c in configs]
    top=sorted(runs,key=lambda r:(-r['validation_f1'],r['trial']))[:3]
    for r in top:
        for seed in [123,2026]:runs.append(train(r['config'],seed,vocab,x,y,out))
    save_json(out/'all_runs.json',runs)
    ranking=[]
    for r in top:
        vals=[v['validation_f1'] for v in runs if v['trial']==r['trial']]
        ranking.append({'trial':r['trial'],'validation_f1_mean':float(np.mean(vals)),'validation_f1_std':float(np.std(vals,ddof=1))})
    ranking.sort(key=lambda r:(-r['validation_f1_mean'],r['trial']))
    winner=ranking[0]['trial'];cp,model=load_model(out/f'{winner}_seed42.pt')
    vp=probabilities(model,x['validation']);vlabels=y['validation'].numpy()
    threshold_rows=[{'threshold':round(float(t),2),**scores(vlabels,vp,t)} for t in np.round(np.arange(.2,.701,.01),2)]
    threshold_rows.sort(key=lambda row:(-row['f1'],abs(row['threshold']-.5),row['threshold']))
    threshold=threshold_rows[0]['threshold']
    selection={'winner':winner,'seed':42,'threshold':threshold,'ranking':ranking,'config':cp['config'],'validation_at_0_5':scores(vlabels,vp),'validation_at_selected_threshold':scores(vlabels,vp,threshold)}
    save_json(out/'selection.json',selection)
    cp['threshold']=threshold;torch.save(cp,out/'mlp_tuned.pt')
    pd.DataFrame(threshold_rows).sort_values('threshold').to_csv(out/'validation_thresholds.csv',index=False)
    # Model/configuration/threshold now fixed. First use of test labels in this script.
    test=pd.read_csv(ROOT/'data/processed/test.csv');tx=encode(test.text.tolist(),vocab,64);ty=test.target.to_numpy()
    tp=probabilities(model,tx)
    baseline_cp,baseline=load_model(ROOT/'artifacts/mlp.pt');bp=probabilities(baseline,tx)
    variants=[('original_mlp',bp,.5),('tuned_mlp_fixed_threshold',tp,.5),('tuned_mlp_validation_threshold',tp,threshold)]
    comparison=[]
    for name,p,t in variants:
        row={'model':name,'threshold':t,**scores(ty,p,t)};comparison.append(row)
        df=test[['id','text','target']].copy();df['probability']=p;df['prediction']=(p>=t).astype(int)
        df.to_csv(out/f'{name}_test_predictions.csv',index=False)
    pd.DataFrame(comparison).to_csv(out/'comparison.csv',index=False)
    # Paired bootstrap conditions on these fitted models; not training-seed uncertainty.
    rng=np.random.default_rng(20260917);deltas=[]
    for _ in range(2000):
        ids=rng.integers(0,len(ty),len(ty));aa=scores(ty[ids],bp[ids]);bb=scores(ty[ids],tp[ids],threshold)
        deltas.append([bb[k]-aa[k] for k in ['accuracy','f1']])
    intervals={k:np.percentile(np.array(deltas)[:,i],[2.5,97.5]).tolist() for i,k in enumerate(['accuracy','f1'])}
    save_json(out/'summary.json',{'comparison':comparison,'selection':selection,'paired_bootstrap_95_percent_delta_intervals':intervals,'bootstrap_caveat':'Tweet-level paired bootstrap conditional on fitted models; event correlation and model-selection/training variability are not represented.','baseline_preserved':all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(ROOT))] for p in paths)})
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(ROOT))] for p in paths)
    print('FINAL COMPARISON',flush=True);print(pd.DataFrame(comparison).to_string(index=False),flush=True)

if __name__=='__main__':main()
