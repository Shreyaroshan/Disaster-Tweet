"""Shared training, validation selection and reproducible artifact format."""
import copy
import json
import platform
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from .utils import ROOT,save_json,sha256
from .preprocessing import load_data,encode
from .embeddings import load_embeddings
from .models import build_model
from .metrics import classification_metrics


SEEDS=[42,123,2026]


def configure_runtime(seed=42):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def probabilities(model,tokens):
    model.eval()
    with torch.no_grad():return torch.cat([model(batch).sigmoid() for batch in tokens.split(128)]).numpy()


def fit(model_name,config,seed,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);configure_runtime(seed)
    parts,vocab,manifest=load_data();weights=load_embeddings(vocab)
    model=build_model(model_name,len(vocab),config,weights)
    tokens={name:encode(frame.text.tolist(),vocab,64) for name,frame in parts.items()}
    y=torch.tensor(parts['train'].target.to_numpy(),dtype=torch.float32)
    embedding_parameters=list(model.embedding.parameters())
    embedding_ids={id(p) for p in embedding_parameters}
    groups=[{'params':[p for p in model.parameters() if id(p) not in embedding_ids and p.requires_grad],'lr':config['learning_rate']}]
    if not config.get('frozen',False):groups.append({'params':embedding_parameters,'lr':config['embedding_learning_rate']})
    optimizer=torch.optim.AdamW(groups,weight_decay=config['weight_decay'])
    loss_function=nn.BCEWithLogitsLoss();rng=np.random.default_rng(seed)
    best=(-1.,-1.);stale=0;history=[];start=time.perf_counter()
    for epoch in range(1,config['epochs']+1):
        model.train();order=rng.permutation(len(y));total=0.
        for begin in range(0,len(order),config['batch_size']):
            idx=order[begin:begin+config['batch_size']];optimizer.zero_grad()
            loss=loss_function(model(tokens['train'][idx]),y[idx]);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();total+=float(loss.detach())*len(idx)
        validation=classification_metrics(parts['validation'].target,probabilities(model,tokens['validation']))
        history.append({'epoch':epoch,'train_loss':total/len(y),**{'validation_'+k:v for k,v in validation.items()}})
        key=(validation['accuracy'],validation['f1'])
        if key>best:best,state,best_epoch,stale=key,copy.deepcopy(model.state_dict()),epoch,0
        else:stale+=1
        if stale>=config['patience']:break
    elapsed=time.perf_counter()-start;model.load_state_dict(state)
    p=probabilities(model,tokens['validation'])
    checkpoint={'format_version':1,'model':model_name,'config':config,'seed':seed,'vocabulary':vocab,'state_dict':state,'threshold':.5,
                'max_length':64,'data_manifest_sha256':sha256(ROOT/'data/processed/manifest.json')}
    torch.save(checkpoint,output/'model.pt')
    pd.DataFrame(history).to_csv(output/'history.csv',index=False)
    pd.DataFrame({'id':parts['validation'].id,'target':parts['validation'].target,'probability':p}).to_csv(output/'validation_predictions.csv',index=False)
    metadata={'model':model_name,'config':config,'seed':seed,'best_epoch':best_epoch,'epochs_run':len(history),'training_seconds':elapsed,
              'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
              'validation':classification_metrics(parts['validation'].target,p),'device':'cpu','threads':4,
              'python':platform.python_version(),'torch':torch.__version__,'checkpoint_sha256':sha256(output/'model.pt')}
    save_json(output/'metadata.json',metadata)
    print(f"{model_name} {output.name}: val accuracy={metadata['validation']['accuracy']:.4f}, F1={metadata['validation']['f1']:.4f}, epoch={best_epoch}, seconds={elapsed:.1f}",flush=True)
    return metadata


def load_checkpoint(path):
    cp=torch.load(path,map_location='cpu',weights_only=True)
    assert cp['format_version']==1
    model=build_model(cp['model'],len(cp['vocabulary']),cp['config']);model.load_state_dict(cp['state_dict']);model.eval()
    return cp,model


def search(model_name,run_name,configurations):
    folder=ROOT/'results'/run_name/model_name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'plan.json').exists():raise FileExistsError(f'Run already started: {folder}')
    plan={'model':model_name,'configurations':configurations,'screen_seed':42,'confirmation_seeds':[123,2026],
          'objective':'Validation accuracy primary; positive-class F1 breaks ties. No selection by test score.',
          'screen':'All predefined configurations at seed42. Top two repeated at seeds123 and2026. Fixed single-model deployment uses seed42.',
          'threshold':'For winning configuration compare seed42 and equal-weight 3-seed ensemble, thresholds0.35..0.65 step.01. Accuracy, F1, proximity.5, single preferred.',
          'test':'Separate evaluation after all four selections saved. Existing reused local test, exploratory comparison.'}
    save_json(folder/'plan.json',plan);records=[]
    for i,config in enumerate(configurations,1):
        name=f'candidate_{i:02}_seed42';r=fit(model_name,config,42,folder/name);r['candidate']=i;r['path']=name;records.append(r);save_json(folder/'runs.json',records)
    top=sorted(records,key=lambda r:(-r['validation']['accuracy'],-r['validation']['f1'],r['candidate']))[:min(2,len(records))]
    for row in top:
        for seed in [123,2026]:
            name=f"candidate_{row['candidate']:02}_seed{seed}";r=fit(model_name,row['config'],seed,folder/name)
            r['candidate']=row['candidate'];r['path']=name;records.append(r);save_json(folder/'runs.json',records)
    ranking=[]
    for row in top:
        values=[r['validation'] for r in records if r['candidate']==row['candidate']]
        ranking.append({'candidate':row['candidate'],'mean_accuracy':float(np.mean([v['accuracy'] for v in values])),
                        'std_accuracy':float(np.std([v['accuracy'] for v in values],ddof=1)),'mean_f1':float(np.mean([v['f1'] for v in values]))})
    ranking.sort(key=lambda r:(-r['mean_accuracy'],-r['mean_f1'],r['candidate']))
    winner=ranking[0]['candidate'];rows=[r for r in records if r['candidate']==winner]
    seed42=next(r for r in rows if r['seed']==42)
    arrays=[pd.read_csv(folder/r['path']/'validation_predictions.csv').probability.to_numpy() for r in rows]
    labels=pd.read_csv(folder/seed42['path']/'validation_predictions.csv').target.to_numpy()
    options=[]
    for ensemble,p in [(False,pd.read_csv(folder/seed42['path']/'validation_predictions.csv').probability.to_numpy()),(True,np.mean(arrays,axis=0))]:
        for threshold in np.round(np.arange(.35,.651,.01),2):options.append({'ensemble':ensemble,'threshold':float(threshold),**classification_metrics(labels,p,threshold)})
    options.sort(key=lambda r:(-r['accuracy'],-r['f1'],abs(r['threshold']-.5),r['ensemble']))
    selected=options[0];chosen=rows if selected['ensemble'] else [seed42]
    selection={'model':model_name,'config':seed42['config'],'ranking':ranking,'threshold':selected['threshold'],'ensemble':selected['ensemble'],
               'validation':selected,'members':[{'path':str((folder/r['path']/'model.pt').relative_to(ROOT)),'sha256':sha256(folder/r['path']/'model.pt'),'seed':r['seed']} for r in chosen],
               'parameters_per_model':seed42['parameters'],'training_seconds':sum(r['training_seconds'] for r in records),
               'selected_fit_seconds':sum(r['training_seconds'] for r in chosen),'search_fits':len(records),'search_history':str(folder.relative_to(ROOT))}
    pd.DataFrame(options).to_csv(folder/'thresholds.csv',index=False)
    save_json(folder/'selection.json',selection)
    print('SELECTED '+model_name+' '+json.dumps(selected),flush=True)
    return selection


def selected_probabilities(selection,texts):
    output=[]
    for member in selection['members']:
        path=ROOT/member['path'];assert sha256(path)==member['sha256']
        cp,model=load_checkpoint(path)
        output.append(probabilities(model,encode(list(texts),cp['vocabulary'],64)))
    return np.mean(output,axis=0)
