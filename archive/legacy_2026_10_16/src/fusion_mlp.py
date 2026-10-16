"""MLP fusion of train-fitted lexical TF-IDF and pooled static word embeddings.

Follow-up after the recorded dual-embedding experiment. Existing test results have
been inspected; all subsequent results are explicitly exploratory. Selection still
uses only CV/validation, and the previous MLP checkpoint is never overwritten.
"""
import argparse
import copy
import json
import time
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from project import ROOT, clean, encode, save_json
from final_mlp import fit_features, preservation, SOURCES
from improve_mlp import metrics, digest

DEFAULT_RUN='mlp_fusion_2026_10_16'
REFERENCE=ROOT/'experiments/mlp_final_2026_10_16'


def configs():
    base=dict(hidden=64,dropout=.5,lr=.001,embedding_lr=.0002,weight_decay=.01,epochs=30,patience=6,batch_size=64)
    changes=[dict(sources=['wiki100'],frozen=False,characters=False),
             dict(sources=['wiki300'],frozen=False,characters=False),
             dict(sources=['wiki300'],frozen=True,characters=False),
             dict(sources=['wiki300'],frozen=True,characters=True)]
    return [dict(base,**change,name=f'fusion_{i:02}') for i,change in enumerate(changes,1)]


def vectorizer(characters):
    word=TfidfVectorizer(preprocessor=clean,lowercase=False,ngram_range=(1,2),min_df=2,max_df=.98,max_features=8000,sublinear_tf=True,dtype=np.float32)
    if not characters: return word
    char=TfidfVectorizer(preprocessor=clean,lowercase=False,analyzer='char_wb',ngram_range=(3,5),min_df=3,max_features=8000,sublinear_tf=True,dtype=np.float32)
    return FeatureUnion([('word',word),('char',char)],transformer_weights={'word':1.,'char':.5})


class FusionMLP(nn.Module):
    def __init__(self,config,vocab_size,lexical_size,weights=None):
        super().__init__()
        dimension=sum(SOURCES[s][1] for s in config['sources'])
        hidden=config['hidden']
        self.embedding=nn.Embedding(vocab_size,dimension,padding_idx=0)
        if weights is not None:
            with torch.no_grad(): self.embedding.weight.copy_(torch.as_tensor(weights))
        self.embedding.weight.requires_grad_(not config['frozen'])
        self.semantic=nn.Sequential(nn.LayerNorm(dimension*2),nn.Dropout(.3),nn.Linear(dimension*2,hidden),nn.ReLU())
        self.lexical=nn.Sequential(nn.Dropout(.3),nn.Linear(lexical_size,hidden),nn.ReLU())
        self.head=nn.Sequential(nn.Dropout(config['dropout']),nn.Linear(2*hidden,hidden),nn.ReLU(),nn.Dropout(config['dropout']),nn.Linear(hidden,1))

    def forward(self,tokens,tfidf):
        mask=tokens.ne(0).unsqueeze(-1)
        embedded=self.embedding(tokens)
        mean=(embedded*mask).sum(1)/mask.sum(1).clamp(min=1)
        maximum=embedded.masked_fill(~mask,-torch.inf).amax(1)
        maximum=torch.where(torch.isfinite(maximum),maximum,torch.zeros_like(maximum))
        semantic=self.semantic(torch.cat((mean,maximum),1))
        lexical=self.lexical(5*tfidf)
        return self.head(torch.cat((semantic,lexical),1)).squeeze(-1)


def probabilities(model,tokens,lexical):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(tokens[i:i+128],torch.from_numpy(lexical[i:i+128].toarray())).sigmoid() for i in range(0,len(tokens),128)]).numpy()


def train(config,seed,fitting,stopping,out,name):
    torch.manual_seed(seed);np.random.seed(seed)
    vocab,weights,_=fit_features(fitting.text.tolist(),config['sources'])
    vec=vectorizer(config['characters'])
    lexical=vec.fit_transform(fitting.text).tocsr();stop_lex=vec.transform(stopping.text).tocsr()
    tokens=encode(fitting.text.tolist(),vocab,64);stop_tokens=encode(stopping.text.tolist(),vocab,64)
    y=torch.tensor(fitting.target.to_numpy(),dtype=torch.float32)
    model=FusionMLP(config,len(vocab),lexical.shape[1],weights)
    groups=[dict(params=list(model.semantic.parameters())+list(model.lexical.parameters())+list(model.head.parameters()),lr=config['lr'])]
    if not config['frozen']: groups.append(dict(params=model.embedding.parameters(),lr=config['embedding_lr']))
    optimizer=torch.optim.AdamW(groups,weight_decay=config['weight_decay'])
    loss_fn=nn.BCEWithLogitsLoss();rng=np.random.default_rng(seed)
    best=(-1.,-1.);stale=0;history=[];start=time.perf_counter()
    for epoch in range(1,config['epochs']+1):
        model.train();order=rng.permutation(len(y));total=0.
        for begin in range(0,len(order),config['batch_size']):
            idx=order[begin:begin+config['batch_size']]
            optimizer.zero_grad()
            loss=loss_fn(model(tokens[idx],torch.from_numpy(lexical[idx].toarray())),y[idx]);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();total+=float(loss.detach())*len(idx)
        score=metrics(stopping.target,probabilities(model,stop_tokens,stop_lex));history.append(dict(epoch=epoch,loss=total/len(y),**score))
        key=(score['accuracy'],score['f1'])
        if key>best: best,state,best_epoch,stale=key,copy.deepcopy(model.state_dict()),epoch,0
        else: stale+=1
        if stale>=config['patience']: break
    model.load_state_dict(state)
    torch.save(dict(config=config,seed=seed,vocabulary=vocab,lexical_size=lexical.shape[1],state_dict=state),out/f'{name}.pt')
    joblib.dump(vec,out/f'{name}_vectorizer.joblib')
    pd.DataFrame(history).to_csv(out/f'{name}_history.csv',index=False)
    record=dict(name=name,config=config,seed=seed,best_epoch=best_epoch,epochs_run=len(history),seconds=time.perf_counter()-start,parameters=sum(p.numel() for p in model.parameters()))
    return model,vocab,vec,record


def search(out):
    out.mkdir(parents=True,exist_ok=False)
    save_json(out/'plan.json',dict(configurations=configs(),seeds=[42,123],
        context='Adaptive follow-up after dual-embedding test accuracy decreased. All outcomes are exploratory; unsuccessful prior experiment is retained.',
        folds='Same three outer and inner folds as mlp_final_2026_10_16. Vocabularies/IDF/TF-IDF fit on inner training only.',
        selection='Four configurations times two seeds times three folds. Choose mean CV accuracy then F1 at fixed 0.5.',
        confirmation='Winning config fitted at seeds42,123,2026; validation selects designated seed42 or equal-weight three-seed ensemble, threshold .35..65 in .01 steps by accuracy then F1, proximity .5, then single.',
        test_policy='Selection fixed before one separate test evaluation. Do not select seed/configuration from test outcomes.',
        reference='Reference architecture CV metrics from preceding experiment; same rows, inner stopping splits and seeds. Optimizer common; fusion maximum30 epochs versus reference40.'))
    save_json(out/'input_hashes.json',preservation())
    frame=pd.read_csv(ROOT/'data/processed/train.csv').set_index('id',drop=False)
    folds=json.loads((REFERENCE/'folds.json').read_text());save_json(out/'folds.json',folds)
    records=[]
    for config in configs():
        for seed in [42,123]:
            for fold,ids in enumerate(folds):
                name=f"{config['name']}_seed{seed}_fold{fold}"
                model,vocab,vec,record=train(config,seed,frame.loc[ids['fitting_ids']],frame.loc[ids['stopping_ids']],out,name)
                held=frame.loc[ids['assessment_ids']]
                p=probabilities(model,encode(held.text.tolist(),vocab,64),vec.transform(held.text).tocsr())
                record.update(fold=fold,assessment=metrics(held.target,p));records.append(record)
                pd.DataFrame(dict(id=held.id.to_numpy(),target=held.target.to_numpy(),probability=p)).to_csv(out/f'{name}_assessment.csv',index=False)
                save_json(out/'runs.json',records)
                print(f"{name}: accuracy={record['assessment']['accuracy']:.4f}, F1={record['assessment']['f1']:.4f}, {record['seconds']:.1f}s",flush=True)
    ranking=[]
    for config in configs():
        selected=[r['assessment'] for r in records if r['config']['name']==config['name']]
        ranking.append(dict(name=config['name'],accuracy=float(np.mean([r['accuracy'] for r in selected])),f1=float(np.mean([r['f1'] for r in selected])),accuracy_std=float(np.std([r['accuracy'] for r in selected],ddof=1))))
    ranking.sort(key=lambda r:(-r['accuracy'],-r['f1'],r['name']))
    save_json(out/'cv_selection.json',dict(winner=ranking[0]['name'],ranking=ranking,config=next(c for c in configs() if c['name']==ranking[0]['name'])))
    assert preservation()==json.loads((out/'input_hashes.json').read_text())
    print('CV COMPLETE\n'+json.dumps(ranking,indent=2),flush=True)


def confirm(out):
    if (out/'confirmation_started.json').exists(): raise FileExistsError('Already started')
    chosen=json.loads((out/'cv_selection.json').read_text())
    save_json(out/'confirmation_started.json',dict(cv_selection_sha256=digest(out/'cv_selection.json')))
    fitting=pd.read_csv(ROOT/'data/processed/train.csv');validation=pd.read_csv(ROOT/'data/processed/validation.csv')
    arrays=[];records=[];seeds=[42,123,2026]
    for seed in seeds:
        name=f'deploy_seed{seed}'
        model,vocab,vec,record=train(chosen['config'],seed,fitting,validation,out,name)
        p=probabilities(model,encode(validation.text.tolist(),vocab,64),vec.transform(validation.text).tocsr());arrays.append(p)
        record['validation']=metrics(validation.target,p);records.append(record)
        np.save(out/f'{name}_validation.npy',p)
        print(f"{name}: {record['validation']}",flush=True)
    save_json(out/'confirmation_runs.json',records)
    candidates=[]
    for ensemble,p in [(False,arrays[0]),(True,np.mean(arrays,axis=0))]:
        for threshold in np.round(np.arange(.35,.651,.01),2): candidates.append(dict(ensemble=ensemble,threshold=float(threshold),**metrics(validation.target,p,threshold)))
    candidates.sort(key=lambda r:(-r['accuracy'],-r['f1'],abs(r['threshold']-.5),r['ensemble']))
    selected=candidates[0];deploy_seeds=seeds if selected['ensemble'] else [42]
    old=json.loads((ROOT/'experiments/mlp_improved_2026_09_18/selection.json').read_text())['validation_selected_threshold']
    files=[f'deploy_seed{seed}{suffix}' for seed in deploy_seeds for suffix in ['.pt','_vectorizer.joblib']]
    selection=dict(config=chosen['config'],seeds=deploy_seeds,threshold=selected['threshold'],ensemble=selected['ensemble'],validation=selected,previous_validation=old,
                   validation_gate_passed=selected['accuracy']>=old['accuracy'] and selected['f1']>=old['f1'],artifact_sha256={f:digest(out/f) for f in files})
    p=np.mean(arrays,axis=0) if selected['ensemble'] else arrays[0]
    pd.DataFrame(dict(id=validation.id,target=validation.target,probability=p)).to_csv(out/'validation_predictions.csv',index=False)
    pd.DataFrame(candidates).to_csv(out/'thresholds.csv',index=False)
    save_json(out/'selection.json',selection)
    print('SELECTION LOCKED\n'+json.dumps(selection,indent=2),flush=True)


def load(out):
    selection=json.loads((out/'selection.json').read_text());models=[]
    for filename,checksum in selection['artifact_sha256'].items(): assert digest(out/filename)==checksum
    for seed in selection['seeds']:
        cp=torch.load(out/f'deploy_seed{seed}.pt',map_location='cpu',weights_only=True)
        model=FusionMLP(cp['config'],len(cp['vocabulary']),cp['lexical_size']);model.load_state_dict(cp['state_dict'])
        # Trust only locally generated, checksum-verified sklearn objects.
        vec=joblib.load(out/f'deploy_seed{seed}_vectorizer.joblib')
        models.append((model.eval(),cp['vocabulary'],vec))
    return selection,models


def predict(loaded,texts):
    texts=list(texts)
    return np.mean([probabilities(model,encode(texts,vocab,64),vec.transform(texts).tocsr()) for model,vocab,vec in loaded[1]],axis=0)


def evaluate(out):
    if (out/'evaluation_started.json').exists(): raise FileExistsError('Already evaluated')
    loaded=load(out);selection=loaded[0]
    assert preservation()==json.loads((out/'input_hashes.json').read_text())
    save_json(out/'evaluation_started.json',dict(selection_sha256=digest(out/'selection.json'),time_unix=time.time()))
    test=pd.read_csv(ROOT/'data/processed/test.csv');p=predict(loaded,test.text)
    rows=test[['id','text','target']].copy();rows['probability']=p;rows['prediction']=(p>=selection['threshold']).astype(int);rows.to_csv(out/'test_predictions.csv',index=False)
    previous=pd.read_csv(ROOT/'experiments/mlp_improved_2026_09_18/test_predictions.csv')
    assert np.array_equal(test.id,previous.id)
    old=metrics(test.target,previous.probability,.49);new=metrics(test.target,p,selection['threshold'])
    comparison=[dict(model='September MLP',**old),dict(model='Lexical-semantic fusion MLP',**new)]
    rng=np.random.default_rng(20261016);deltas=[];y=test.target.to_numpy();bp=previous.probability.to_numpy()
    for _ in range(2000):
        idx=rng.integers(0,len(y),len(y));a=metrics(y[idx],bp[idx],.49);b=metrics(y[idx],p[idx],selection['threshold'])
        deltas.append([b[k]-a[k] for k in ['accuracy','f1']])
    intervals={k:np.percentile(np.array(deltas)[:,i],[2.5,97.5]).tolist() for i,k in enumerate(['accuracy','f1'])}
    save_json(out/'summary.json',dict(comparison=comparison,improvement={k:new[k]-old[k] for k in new},paired_bootstrap_95_percent_delta_intervals=intervals,
        validation_gate_passed=selection['validation_gate_passed'],previous_models_and_data_preserved=preservation()==json.loads((out/'input_hashes.json').read_text()),
        caveat='Adaptive follow-up after inspecting earlier test outcomes. Selection used CV/validation, but reused test is exploratory and cannot establish independent generalization.'))
    pd.DataFrame(comparison).to_csv(out/'comparison.csv',index=False)
    kaggle=pd.read_csv(ROOT/'data/raw/test.csv');pd.DataFrame(dict(id=kaggle.id,target=(predict(loaded,kaggle.text)>=selection['threshold']).astype(int))).to_csv(out/'kaggle_submission.csv',index=False)
    print(pd.DataFrame(comparison).to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['search','confirm','evaluate','predict']);parser.add_argument('--run-name',default=DEFAULT_RUN);parser.add_argument('--text');args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);out=ROOT/'experiments'/args.run_name
    if args.command=='search': search(out)
    elif args.command=='confirm': confirm(out)
    elif args.command=='evaluate': evaluate(out)
    else:
        if args.text is None: parser.error('--text required')
        loaded=load(out);p=float(predict(loaded,[args.text])[0]);print(json.dumps(dict(disaster_probability=p,threshold=loaded[0]['threshold'],prediction=int(p>=loaded[0]['threshold'])),indent=2))


if __name__=='__main__': main()
