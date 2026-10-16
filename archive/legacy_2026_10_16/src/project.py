"""Reproducible, text-only MLP and CNN disaster tweet experiment."""
import argparse, copy, hashlib, html, json, os, random, re, time, subprocess
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CONFIG = dict(seed=42, max_length=64, max_vocab=20000, embedding_dim=128,
              hidden_dim=64, filters=64, kernels=[3,4,5], dropout=0.4,
              batch_size=64, learning_rate=0.001, epochs=20, patience=4)

def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')

def clean(text):
    text = html.unescape(str(text)).lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' urltoken ', text)
    text = re.sub(r'@\w+', ' usertoken ', text)
    return ' '.join(re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text))

def download():
    import requests
    raw = ROOT/'data/raw'; raw.mkdir(parents=True, exist_ok=True)
    base = 'https://raw.githubusercontent.com/juliuswiscmsba/NLP-Disaster-Tweets/ebb1103dbbb88a30415d7e22f5136018bb119d88/'
    manifest = {'original_source': 'https://www.kaggle.com/competitions/nlp-getting-started/data',
                'acquisition': 'Public mirror of Kaggle competition files; not authenticated directly with Kaggle.', 'files': {}}
    for name, count in [('train.csv',7613), ('test.csv',3263)]:
        path=raw/name
        if not path.exists():
            temporary=path.with_suffix('.download')
            try:
                response=requests.get(base+name,timeout=30); response.raise_for_status()
                temporary.write_bytes(response.content)
            except requests.RequestException:
                subprocess.run(['curl','--fail','--location','--max-time','90',base+name,'--output',str(temporary)],check=True)
            temporary.replace(path)
        df=pd.read_csv(path)
        required={'id','text','keyword','location'} | ({'target'} if name=='train.csv' else set())
        assert required.issubset(df.columns) and len(df)==count, 'Unexpected dataset schema or row count'
        assert df.id.is_unique and df.text.notna().all()
        if name=='train.csv':
            assert df.target.value_counts().to_dict()=={0:4342,1:3271}
        manifest['files'][name]={'url':base+name,'rows':len(df),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    save_json(raw/'provenance.json',manifest)
    print('Downloaded/validated 7,613 labeled and 3,263 unlabeled tweets.',flush=True)

def prepare(config):
    df=pd.read_csv(ROOT/'data/raw/train.csv'); df['clean_text']=df.text.map(clean)
    conflicts=df.groupby('clean_text').target.nunique()
    conflict_keys=set(conflicts[conflicts>1].index)
    rejected=df[df.clean_text.isin(conflict_keys) | df.clean_text.eq('')]
    eligible=df[~df.index.isin(rejected.index)].copy()
    # One representative per normalized tweet prevents identical-input leakage.
    unique=eligible.drop_duplicates('clean_text').copy()
    train,rest=train_test_split(unique,test_size=.30,stratify=unique.target,random_state=config['seed'])
    val,test=train_test_split(rest,test_size=.50,stratify=rest.target,random_state=config['seed'])
    processed=ROOT/'data/processed'; processed.mkdir(parents=True,exist_ok=True)
    rejected.to_csv(processed/'excluded_conflicts_or_empty.csv',index=False)
    splits={'train':train,'validation':val,'test':test}
    for name,part in splits.items(): part.to_csv(processed/f'{name}.csv',index=False)
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not set(splits[a].clean_text)&set(splits[b].clean_text)
    counts=Counter(' '.join(train.clean_text).split())
    words=sorted(counts,key=lambda w:(-counts[w],w))[:config['max_vocab']-2]
    vocab={'<PAD>':0,'<UNK>':1,**{w:i+2 for i,w in enumerate(words)}}
    save_json(processed/'vocabulary.json',vocab)
    eda={'raw_rows':len(df),'raw_class_counts':{str(k):int(v) for k,v in df.target.value_counts().items()},
         'missing_metadata':df[['keyword','location']].isna().sum().to_dict(),
         'excluded_conflicting_or_empty_rows':len(rejected),'conflicting_normalized_texts':len(conflict_keys),
         'removed_same_label_duplicates':len(eligible)-len(unique),'retained_rows':len(unique),
         'vocabulary_size':len(vocab),'splits':{k:{'rows':len(v),'positive':int(v.target.sum())} for k,v in splits.items()},
         'train_length_percentiles':{str(p):float(np.percentile(train.clean_text.str.split().str.len(),p)) for p in [50,90,95,99,100]},
         'raw_sha256':hashlib.sha256((ROOT/'data/raw/train.csv').read_bytes()).hexdigest()}
    save_json(ROOT/'reports/eda.json',eda)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    df.target.value_counts().sort_index().plot.bar(ax=axes[0],color=['#64748b','#e6793b'])
    axes[0].set(title='Raw class distribution',xlabel='0 = non-disaster; 1 = disaster',ylabel='Tweets')
    axes[1].hist(train.clean_text.str.split().str.len(),bins=30,color='#287c8e')
    axes[1].set(title='Training tweet lengths',xlabel='Tokens',ylabel='Tweets')
    fig.tight_layout(); fig.savefig(ROOT/'reports/eda.png',dpi=160); plt.close(fig)
    return splits,vocab

def encode(texts,vocab,length):
    arr=np.zeros((len(texts),length),dtype=np.int64)
    for i,text in enumerate(texts):
        ids=[vocab.get(t,1) for t in clean(text).split()][:length] or [1]
        arr[i,:len(ids)]=ids
    return torch.from_numpy(arr)

class MLP(nn.Module):
    """Trainable embedding -> masked mean -> feed-forward hidden layer -> logit."""
    def __init__(self,n,c):
        super().__init__(); self.embedding=nn.Embedding(n,c['embedding_dim'],padding_idx=0)
        self.head=nn.Sequential(nn.Linear(c['embedding_dim'],c['hidden_dim']),nn.ReLU(),nn.Dropout(c['dropout']),nn.Linear(c['hidden_dim'],1))
    def forward(self,x):
        mask=(x!=0).unsqueeze(-1)
        mean=(self.embedding(x)*mask).sum(1)/mask.sum(1).clamp(min=1)
        return self.head(mean).squeeze(-1)

class CNN(nn.Module):
    """Parallel 1D convolution filters -> masked global max pooling -> logit."""
    def __init__(self,n,c):
        super().__init__(); self.embedding=nn.Embedding(n,c['embedding_dim'],padding_idx=0)
        self.convs=nn.ModuleList([nn.Conv1d(c['embedding_dim'],c['filters'],k) for k in c['kernels']])
        self.head=nn.Sequential(nn.Dropout(c['dropout']),nn.Linear(c['filters']*len(c['kernels']),1))
    def forward(self,x):
        e=self.embedding(x).transpose(1,2); lengths=(x!=0).sum(1); pooled=[]
        for conv in self.convs:
            features=torch.relu(conv(e)); k=conv.kernel_size[0]
            valid=(lengths-k+1).clamp(min=1)
            mask=torch.arange(features.shape[2],device=x.device)[None,:] < valid[:,None]
            pooled.append(features.masked_fill(~mask[:,None,:],float('-inf')).amax(2))
        return self.head(torch.cat(pooled,dim=1)).squeeze(-1)

MODELS={'mlp':MLP,'cnn':CNN}

def metrics(y,p):
    predicted=np.asarray(p)>=.5
    return {'accuracy':float(accuracy_score(y,predicted)), 'precision':float(precision_score(y,predicted,zero_division=0)),
            'recall':float(recall_score(y,predicted,zero_division=0)), 'f1':float(f1_score(y,predicted,zero_division=0))}

def probabilities(model,x,batch=256):
    model.eval()
    with torch.no_grad(): return torch.cat([model(z).sigmoid() for z in x.split(batch)]).numpy()

def train_all(config):
    download(); splits,vocab=prepare(config)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    tensors={k:encode(v.text.tolist(),vocab,config['max_length']) for k,v in splits.items()}
    labels={k:torch.tensor(v.target.to_numpy(),dtype=torch.float32) for k,v in splits.items()}
    results=[]; out=ROOT/'artifacts'; out.mkdir(exist_ok=True)
    save_json(out/'config.json',config)
    for name,cls in MODELS.items():
        random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
        model=cls(len(vocab),config)
        loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tensors['train'],labels['train']),batch_size=config['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(config['seed']))
        optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
        loss_fn=nn.BCEWithLogitsLoss(); best=-1.; stale=0; history=[]; start=time.perf_counter()
        for epoch in range(1,config['epochs']+1):
            model.train(); total=0
            for x,y in loader:
                optimizer.zero_grad(); loss=loss_fn(model(x),y); loss.backward(); optimizer.step(); total+=loss.item()*len(y)
            p=probabilities(model,tensors['validation']); score=metrics(labels['validation'].numpy(),p)
            row={'epoch':epoch,'train_loss':total/len(labels['train']),**{'validation_'+k:v for k,v in score.items()}}
            history.append(row); print(name,row,flush=True)
            if score['f1']>best:
                best=score['f1']; best_state=copy.deepcopy(model.state_dict()); best_epoch=epoch; stale=0
            else: stale+=1
            if stale>=config['patience']: break
        elapsed=time.perf_counter()-start; model.load_state_dict(best_state)
        torch.save({'model':name,'config':config,'vocabulary':vocab,'state_dict':best_state},out/f'{name}.pt')
        pd.DataFrame(history).to_csv(ROOT/f'reports/{name}_history.csv',index=False)
        p=probabilities(model,tensors['test']); score=metrics(labels['test'].numpy(),p)
        results.append({'model':name,**score,'validation_f1':best,'best_epoch':best_epoch,'epochs_run':len(history),'training_seconds':elapsed,'parameters':sum(t.numel() for t in model.parameters()),'device':'cpu'})
        predicted=splits['test'][['id','text','target']].copy(); predicted['probability']=p; predicted['prediction']=(p>=.5).astype(int)
        predicted.to_csv(ROOT/f'reports/{name}_test_predictions.csv',index=False)
        errors=predicted[predicted.target!=predicted.prediction].copy(); errors['review_category']=''; errors['review_notes']=''
        errors.to_csv(ROOT/f'reports/{name}_errors.csv',index=False)
        fig,ax=plt.subplots(figsize=(4.5,4)); cm=confusion_matrix(predicted.target,predicted.prediction,labels=[0,1]); ax.imshow(cm,cmap='Blues')
        for i in range(2):
            for j in range(2): ax.text(j,i,str(cm[i,j]),ha='center',va='center')
        ax.set(xticks=[0,1],yticks=[0,1],xlabel='Predicted label',ylabel='True label',title=f'{name.upper()} held-out confusion matrix')
        fig.tight_layout(); fig.savefig(ROOT/f'reports/{name}_confusion.png',dpi=160); plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(9,3.5)); h=pd.DataFrame(history)
        axes[0].plot(h.epoch,h.train_loss); axes[0].set(title=f'{name.upper()} training loss',xlabel='Epoch',ylabel='BCE')
        axes[1].plot(h.epoch,h.validation_f1); axes[1].set(title='Validation F1',xlabel='Epoch',ylabel='F1')
        fig.tight_layout(); fig.savefig(ROOT/f'reports/{name}_learning.png',dpi=160); plt.close(fig)
        competition=pd.read_csv(ROOT/'data/raw/test.csv'); cp=probabilities(model,encode(competition.text.tolist(),vocab,config['max_length']))
        pd.DataFrame({'id':competition.id,'target':(cp>=.5).astype(int)}).to_csv(out/f'{name}_kaggle_submission.csv',index=False)
    pd.DataFrame(results).to_csv(ROOT/'reports/comparison.csv',index=False)
    save_json(ROOT/'reports/results.json',results)
    print(pd.DataFrame(results).to_string(index=False),flush=True)

def predict(name,text):
    checkpoint=torch.load(ROOT/f'artifacts/{name}.pt',map_location='cpu',weights_only=True)
    model=MODELS[name](len(checkpoint['vocabulary']),checkpoint['config']); model.load_state_dict(checkpoint['state_dict'])
    p=float(probabilities(model,encode([text],checkpoint['vocabulary'],checkpoint['config']['max_length']))[0])
    return {'model':name,'disaster_probability':p,'prediction':int(p>=.5),'label':'disaster' if p>=.5 else 'non-disaster'}

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['download','train','predict']); parser.add_argument('--model',choices=list(MODELS),default='cnn'); parser.add_argument('--text'); parser.add_argument('--epochs',type=int,default=20)
    args=parser.parse_args()
    if args.command=='download': download()
    elif args.command=='train': train_all({**CONFIG,'epochs':args.epochs})
    else:
        if not args.text: parser.error('--text is required for predict')
        print(json.dumps(predict(args.model,args.text),indent=2))
