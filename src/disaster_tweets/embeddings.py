"""Verified, training-vocabulary-only GloVe 6B 100d preparation."""
import gzip
import hashlib
import json
import subprocess
import numpy as np
from .utils import ROOT,save_json,sha256


def prepare_embeddings():
    folder=ROOT/'data/embeddings';folder.mkdir(parents=True,exist_ok=True)
    path=folder/'glove-wiki-gigaword-100.gz'
    url='https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz'
    if not path.exists():
        temporary=path.with_suffix('.gz.download')
        subprocess.run(['curl','-fL','--retry','2','--max-time','900',url,'-o',str(temporary)],check=True)
        temporary.replace(path)
    h=hashlib.md5()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    assert h.hexdigest()=='40ec481866001177b8cd4cb0df92924f'
    vocabulary_path=ROOT/'data/processed/vocabulary.json'
    vocabulary=json.loads(vocabulary_path.read_text())
    matrix_path=folder/'glove_train_vocabulary.npy'
    provenance_path=folder/'provenance.json'
    if matrix_path.exists() and provenance_path.exists():
        provenance=json.loads(provenance_path.read_text())
        assert provenance['training_vocabulary_sha256']==sha256(vocabulary_path)
        assert provenance['matrix_sha256']==sha256(matrix_path)
        return provenance
    matrix=np.random.default_rng(42).normal(0,.05,(len(vocabulary),100)).astype('float32')
    found=set()
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        assert stream.readline().strip()=='400000 100'
        for line in stream:
            word,_,values=line.rstrip().partition(' ')
            if word in vocabulary:
                vector=np.fromstring(values,sep=' ',dtype=np.float32);assert len(vector)==100
                matrix[vocabulary[word]]=vector;found.add(word)
    matrix[0]=0;matrix[1]=np.mean(matrix[[vocabulary[w] for w in sorted(found)]],axis=0)
    np.save(matrix_path,matrix)
    provenance={'source':url,'original_source':'https://nlp.stanford.edu/projects/glove/','license':'https://opendatacommons.org/licenses/pddl/1-0/',
                'sha256':sha256(path),'training_vocabulary_sha256':sha256(vocabulary_path),'matrix_sha256':sha256(matrix_path),
                'matched_training_words':len(found),'training_vocabulary_words':len(vocabulary)-2}
    save_json(provenance_path,provenance);return provenance


def load_embeddings(vocabulary):
    folder=ROOT/'data/embeddings'
    provenance=json.loads((folder/'provenance.json').read_text())
    assert provenance['training_vocabulary_sha256']==sha256(ROOT/'data/processed/vocabulary.json')
    assert provenance['matrix_sha256']==sha256(folder/'glove_train_vocabulary.npy')
    matrix=np.load(folder/'glove_train_vocabulary.npy')
    assert matrix.shape==(len(vocabulary),100) and np.isfinite(matrix).all()
    return matrix
