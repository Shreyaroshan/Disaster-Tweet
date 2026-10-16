"""The single cleaning, split, vocabulary and encoding implementation for all models."""
import html
import json
import re
from collections import Counter
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from .utils import ROOT,save_json,sha256


def clean_text(text):
    text=html.unescape(str(text)).lower()
    text=re.sub(r'https?://\S+|www\.\S+',' urltoken ',text)
    text=re.sub(r'@\w+',' usertoken ',text)
    return ' '.join(re.findall(r"[a-z0-9]+(?:'[a-z]+)?",text))


def fit_vocabulary(texts,max_vocab=20000):
    counts=Counter(' '.join(clean_text(text) for text in texts).split())
    words=sorted(counts,key=lambda word:(-counts[word],word))[:max_vocab-2]
    return {'<PAD>':0,'<UNK>':1,**{word:i+2 for i,word in enumerate(words)}}


def encode(texts,vocabulary,max_length=64):
    output=np.zeros((len(texts),max_length),dtype=np.int64)
    for i,text in enumerate(texts):
        tokens=[vocabulary.get(word,1) for word in clean_text(text).split()][:max_length] or [1]
        output[i,:len(tokens)]=tokens
    return torch.from_numpy(output)


def download_dataset():
    import subprocess
    raw=ROOT/'data/raw';raw.mkdir(parents=True,exist_ok=True)
    base='https://raw.githubusercontent.com/juliuswiscmsba/NLP-Disaster-Tweets/ebb1103dbbb88a30415d7e22f5136018bb119d88/'
    for name in ['train.csv','test.csv']:
        if not (raw/name).exists():
            temporary=raw/(name+'.download')
            subprocess.run(['curl','-fL','--retry','2','--max-time','180',base+name,'-o',str(temporary)],check=True)
            temporary.replace(raw/name)
    manifest={'original_source':'https://www.kaggle.com/competitions/nlp-getting-started/data','acquisition':'Pinned public mirror; not authenticated Kaggle API download.',
              'files':{name:{'url':base+name,'sha256':sha256(raw/name),'rows':len(pd.read_csv(raw/name))} for name in ['train.csv','test.csv']}}
    if not (raw/'provenance.json').exists():save_json(raw/'provenance.json',manifest)


def prepare_data():
    """Create missing partitions deterministically; validate existing partitions without rewriting."""
    download_dataset()
    raw=pd.read_csv(ROOT/'data/raw/train.csv')
    assert len(raw)==7613 and raw.id.is_unique and raw.text.notna().all()
    assert raw.target.value_counts().to_dict()=={0:4342,1:3271}
    raw['clean_text']=raw.text.map(clean_text)
    conflicts=raw.groupby('clean_text').target.nunique()
    conflicting=set(conflicts[conflicts>1].index)
    rejected=raw[raw.clean_text.isin(conflicting)|raw.clean_text.eq('')]
    eligible=raw.drop(rejected.index)
    unique=eligible.drop_duplicates('clean_text')
    train,rest=train_test_split(unique,test_size=.3,stratify=unique.target,random_state=42)
    validation,test=train_test_split(rest,test_size=.5,stratify=rest.target,random_state=42)
    expected={'train':train,'validation':validation,'test':test}
    folder=ROOT/'data/processed';folder.mkdir(parents=True,exist_ok=True)
    for split,frame in expected.items():
        path=folder/f'{split}.csv'
        if not path.exists():frame.to_csv(path,index=False)
        actual=pd.read_csv(path)
        assert actual.id.tolist()==frame.id.tolist(),f'{split} differs from shared split'
        assert actual.target.tolist()==frame.target.tolist()
        assert actual.text.map(clean_text).tolist()==frame.clean_text.tolist()
    vocab=fit_vocabulary(train.text.tolist())
    path=folder/'vocabulary.json'
    if path.exists():assert json.loads(path.read_text())==vocab
    else:save_json(path,vocab)
    if not (folder/'excluded_conflicts_or_empty.csv').exists():rejected.to_csv(folder/'excluded_conflicts_or_empty.csv',index=False)
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not set(expected[a].clean_text)&set(expected[b].clean_text)
        assert not set(expected[a].id)&set(expected[b].id)
    manifest={'seed':42,'raw_rows':len(raw),'retained_rows':len(unique),'excluded_conflicting_or_empty_rows':len(rejected),
              'conflicting_normalized_texts':len(conflicting),'removed_same_label_duplicates':len(eligible)-len(unique),
              'vocabulary_size':len(vocab),'max_length':64,'vocabulary_fit_split':'train',
              'files':{f'{split}.csv':sha256(folder/f'{split}.csv') for split in expected},
              'vocabulary_sha256':sha256(path),'raw_sha256':sha256(ROOT/'data/raw/train.csv'),
              'splits':{name:{'rows':len(frame),'positive':int(frame.target.sum())} for name,frame in expected.items()},
              'cleaning':'HTML decode, lowercase, URL/user placeholders, preserve negation and hashtag words; no stopword removal or stemming.'}
    save_json(folder/'manifest.json',manifest)
    return manifest


def load_data(include_test=False):
    manifest=json.loads((ROOT/'data/processed/manifest.json').read_text())
    names=['train','validation']+(['test'] if include_test else [])
    parts={}
    for name in names:
        path=ROOT/f'data/processed/{name}.csv'
        assert sha256(path)==manifest['files'][f'{name}.csv']
        parts[name]=pd.read_csv(path)
    path=ROOT/'data/processed/vocabulary.json'
    assert sha256(path)==manifest['vocabulary_sha256']
    return parts,json.loads(path.read_text()),manifest
