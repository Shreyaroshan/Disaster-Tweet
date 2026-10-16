"""Final-project MLP: fold-local features, inner early stopping, and CV selection.

Search reads only the original training split. Confirm uses the original validation
split. Evaluate is separate, after model and threshold selection has been locked.
"""
import argparse
import copy
import gzip
import json
import random
import time
from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from project import ROOT, clean, encode, save_json
from improve_mlp import digest, metrics

DEFAULT_RUN = 'mlp_final_2026_10_16'
SOURCES = {
    'wiki100': ('glove-wiki-gigaword-100', 100, '40ec481866001177b8cd4cb0df92924f'),
    'wiki300': ('glove-wiki-gigaword-300', 300, '29e9329ac2241937d55b852e8284e89b'),
    'twitter100': ('glove-twitter-100', 100, 'b04f7bed38756d64cf55b58ce7e97b15'),
}


def prepare(only=None):
    """Extract externally pretrained vectors for original training words only."""
    folder = ROOT/'data/embeddings'
    vocabulary = json.loads((ROOT/'data/processed/vocabulary.json').read_text())
    for source, (name, dimension, checksum) in SOURCES.items():
        if only is not None and source != only:
            continue
        path = folder/f'{name}.gz'
        if not path.exists():
            partial = folder/f'{name}.gz.part'
            if digest(partial, 'md5') != checksum:
                raise ValueError(f'Incomplete or invalid download: {name}')
            partial.rename(path)
        if digest(path, 'md5') != checksum:
            raise ValueError(f'Invalid download: {name}')
        matrix = np.random.default_rng(42).normal(0, .05, (len(vocabulary), dimension)).astype('float32')
        matched = np.zeros(len(vocabulary), dtype=bool)
        aliases = {'<url>': 'urltoken', '<user>': 'usertoken'} if source == 'twitter100' else {}
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            header = handle.readline().split()
            assert int(header[1]) == dimension
            for line in handle:
                word, _, values = line.rstrip().partition(' ')
                word = aliases.get(word, word)
                if word in vocabulary:
                    vector = np.fromstring(values, sep=' ', dtype=np.float32)
                    assert len(vector) == dimension
                    matrix[vocabulary[word]] = vector
                    matched[vocabulary[word]] = True
        matrix[0] = 0
        matrix[1] = matrix[matched].mean(0)
        np.save(folder/f'final_{source}.npy', matrix)
        np.save(folder/f'final_{source}_matched.npy', matched)
        save_json(folder/f'final_{source}_provenance.json', dict(
            source=f'https://github.com/RaRe-Technologies/gensim-data/releases/download/{name}/{name}.gz',
            original='https://nlp.stanford.edu/projects/glove/', license='https://opendatacommons.org/licenses/pddl/1-0/',
            md5=checksum, sha256=digest(path), dimension=dimension, words_matched=int(matched.sum()),
            aliases=aliases, vocabulary_sha256=digest(ROOT/'data/processed/vocabulary.json'),
            matrix_sha256=digest(folder/f'final_{source}.npy'), matched_sha256=digest(folder/f'final_{source}_matched.npy')))
        print(f'{source}: {matched.sum()}/{len(vocabulary)-2} training words matched', flush=True)


def vocabulary_from(texts):
    counts = Counter(' '.join(map(clean, texts)).split())
    words = sorted(counts, key=lambda w: (-counts[w], w))[:19998]
    return {'<PAD>':0, '<UNK>':1, **{word:i+2 for i,word in enumerate(words)}}


def fit_features(texts, sources):
    vocab = vocabulary_from(texts)
    global_vocab = json.loads((ROOT/'data/processed/vocabulary.json').read_text())
    indices = [global_vocab[word] for word in vocab]
    pieces = []
    for source in sources:
        folder = ROOT/'data/embeddings'
        provenance = json.loads((folder/f'final_{source}_provenance.json').read_text())
        assert provenance['vocabulary_sha256'] == digest(ROOT/'data/processed/vocabulary.json')
        assert provenance['matrix_sha256'] == digest(folder/f'final_{source}.npy')
        assert provenance['matched_sha256'] == digest(folder/f'final_{source}_matched.npy')
        matrix = np.load(folder/f'final_{source}.npy')[indices].copy()
        matched = np.load(folder/f'final_{source}_matched.npy')[indices]
        matrix[0] = 0
        # Recompute UNK using only words in this fold's fitting data.
        matrix[1] = matrix[matched].mean(0)
        pieces.append(matrix)
    document_counts = Counter()
    for text in texts:
        document_counts.update(set(clean(text).split()))
    idf = np.ones(len(vocab), dtype=np.float32)
    for word, index in vocab.items():
        if index > 1:
            idf[index] = np.log((1+len(texts))/(1+document_counts[word]))+1
    idf[0] = 0
    return vocab, np.concatenate(pieces, axis=1), idf


class FinalMLP(nn.Module):
    def __init__(self, config, vocab_size, weights=None, idf=None):
        super().__init__()
        self.config = config
        dimension = sum(SOURCES[source][1] for source in config['sources'])
        self.embedding = nn.Embedding(vocab_size, dimension, padding_idx=0)
        if weights is not None:
            with torch.no_grad():
                self.embedding.weight.copy_(torch.as_tensor(weights))
        self.register_buffer('idf', torch.as_tensor(idf) if idf is not None else torch.ones(vocab_size))
        self.head = nn.Sequential(nn.LayerNorm(2*dimension), nn.Dropout(config['dropout']),
                                  nn.Linear(2*dimension, config['hidden']), nn.ReLU(),
                                  nn.Dropout(config['dropout']), nn.Linear(config['hidden'],1))

    def forward(self, tokens):
        mask = tokens.ne(0).unsqueeze(-1)
        embedded = self.embedding(tokens)
        weights = mask.float()
        if self.config['idf_pool']:
            weights = weights * self.idf[tokens].unsqueeze(-1)
        mean = (embedded*weights).sum(1)/weights.sum(1).clamp(min=1)
        maximum = embedded.masked_fill(~mask, -torch.inf).amax(1)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return self.head(torch.cat((mean, maximum),1)).squeeze(-1)


def probabilities(model, x):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(batch).sigmoid() for batch in x.split(256)]).numpy()


def configs():
    base = dict(hidden=128, dropout=.5, idf_pool=False, lr=.001, embedding_lr=.0002,
                weight_decay=.01, epochs=40, patience=6, batch_size=64)
    variants = [dict(sources=['wiki100']), dict(sources=['wiki300']), dict(sources=['twitter100']),
                dict(sources=['wiki300','twitter100']), dict(sources=['wiki300'],idf_pool=True),
                dict(sources=['twitter100'],idf_pool=True), dict(sources=['wiki300','twitter100'],idf_pool=True),
                dict(sources=['wiki300'],hidden=64,dropout=.3)]
    return [dict(base, **variant, name=f'candidate_{i:02}') for i,variant in enumerate(variants,1)]


def train(config, seed, fitting, stopping, out, name):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    vocab, weights, idf = fit_features(fitting.text.tolist(), config['sources'])
    x = encode(fitting.text.tolist(), vocab, 64)
    sx = encode(stopping.text.tolist(), vocab, 64)
    y = torch.tensor(fitting.target.to_numpy(), dtype=torch.float32)
    model = FinalMLP(config, len(vocab), weights, idf)
    optimizer = torch.optim.AdamW([{'params':model.embedding.parameters(),'lr':config['embedding_lr']},
                                    {'params':model.head.parameters(),'lr':config['lr']}], weight_decay=config['weight_decay'])
    loss_function = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    history, best_key, stale = [], (-1.,-1.), 0
    start = time.perf_counter()
    for epoch in range(1,config['epochs']+1):
        model.train()
        order = rng.permutation(len(y))
        total = 0.
        for begin in range(0,len(order),config['batch_size']):
            idx=order[begin:begin+config['batch_size']]
            optimizer.zero_grad()
            loss=loss_function(model(x[idx]),y[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            total += float(loss.detach())*len(idx)
        result=metrics(stopping.target,probabilities(model,sx))
        history.append(dict(epoch=epoch,train_loss=total/len(y),**result))
        key=(result['accuracy'],result['f1'])
        if key>best_key:
            best_key,best_state,best_epoch,stale=key,copy.deepcopy(model.state_dict()),epoch,0
        else:
            stale+=1
        if stale>=config['patience']:
            break
    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(out/f'{name}_history.csv',index=False)
    checkpoint=dict(config=config,seed=seed,vocabulary=vocab,state_dict=best_state,best_epoch=best_epoch)
    torch.save(checkpoint,out/f'{name}.pt')
    metadata=dict(name=name,config=config,seed=seed,best_epoch=best_epoch,epochs_run=len(history),
                  train_rows=len(fitting),stopping_rows=len(stopping),seconds=time.perf_counter()-start,
                  vocabulary_size=len(vocab),parameters=sum(p.numel() for p in model.parameters()))
    return model,vocab,metadata


def preservation():
    paths=['data/processed/train.csv','data/processed/validation.csv','data/processed/test.csv',
           'data/processed/vocabulary.json','artifacts/mlp.pt','artifacts/cnn.pt','reports/results.json',
           'experiments/mlp_improved_2026_09_18/selection.json','experiments/mlp_improved_2026_09_18/trial_15_seed42.pt']
    return {p:digest(ROOT/p) for p in paths if (ROOT/p).exists()}


def search(out):
    out.mkdir(parents=True,exist_ok=False)
    options=configs()
    save_json(out/'plan.json',dict(configurations=options,
        objective='Accuracy primary; disaster-class F1 tie-breaker. All CV evaluation at threshold 0.5.',
        cv='Three stratified outer folds on original train only, random_state=20261016. Within each fitting fold, stratified 15% inner stopping split. Vocabulary, UNK and IDF fit on inner training only.',
        screening='Eight candidates, seed42, three folds. Top two candidates plus candidate_01 reference get seed123 repeat on the same folds. Final rank by mean accuracy over both seeds; then F1.',
        confirmation='Winning configuration trained on original train at seeds 42,123,2026,7,99 with original validation for stopping. Compare seed42 versus five-seed equal-weight ensemble; threshold .35..65 step .01, validation accuracy then F1, proximity .5, single preferred.',
        validation_gate='Recommend replacement only if final validation accuracy and F1 both match or exceed the saved current MLP. Record gate before test.',
        evaluation='One final evaluation on existing reused test. Selection cannot change after evaluation. CV is used for selection, not claimed to be a fully independent nested-CV performance estimate.',
        scope='MLP classifiers over static pretrained word embeddings; no Transformer, CNN or LSTM.'))
    save_json(out/'input_hashes.json',preservation())
    for source in SOURCES:
        save_json(out/f'{source}_provenance.json',json.loads((ROOT/f'data/embeddings/final_{source}_provenance.json').read_text()))
    frame=pd.read_csv(ROOT/'data/processed/train.csv').reset_index(drop=True)
    folds=[]
    for fold,(development,assessment) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261016).split(frame,frame.target)):
        fitting,stopping=train_test_split(development,test_size=.15,stratify=frame.target.iloc[development],random_state=20261016+fold)
        folds.append((fitting,stopping,assessment))
    save_json(out/'folds.json',[{label:frame.id.iloc[idx].tolist() for label,idx in zip(['fitting_ids','stopping_ids','assessment_ids'],fold)} for fold in folds])
    records=[]
    def run_config(config,seed):
        for fold,(fitting,stopping,assessment) in enumerate(folds):
            name=f"{config['name']}_seed{seed}_fold{fold}"
            model,vocab,record=train(config,seed,frame.iloc[fitting],frame.iloc[stopping],out,name)
            p=probabilities(model,encode(frame.text.iloc[assessment].tolist(),vocab,64))
            rows=frame.iloc[assessment][['id','target']].copy()
            rows['probability']=p
            rows.to_csv(out/f'{name}_assessment.csv',index=False)
            record.update(fold=fold,assessment=metrics(rows.target,p))
            records.append(record)
            save_json(out/'runs.json',records)
            print(f"{name}: CV accuracy={record['assessment']['accuracy']:.4f}, F1={record['assessment']['f1']:.4f}, best epoch={record['best_epoch']}, {record['seconds']:.1f}s",flush=True)
    for config in options:
        run_config(config,42)
    screening=[]
    for config in options:
        selected=[r['assessment'] for r in records if r['config']['name']==config['name']]
        screening.append(dict(name=config['name'],accuracy=float(np.mean([r['accuracy'] for r in selected])),f1=float(np.mean([r['f1'] for r in selected]))))
    screening.sort(key=lambda r:(-r['accuracy'],-r['f1'],r['name']))
    finalists=sorted(set([r['name'] for r in screening[:2]]+['candidate_01']))
    save_json(out/'screening.json',dict(ranking=screening,finalists=finalists))
    for config in options:
        if config['name'] in finalists:
            run_config(config,123)
    ranking=[]
    for name in finalists:
        selected=[r['assessment'] for r in records if r['config']['name']==name]
        ranking.append(dict(name=name,accuracy=float(np.mean([r['accuracy'] for r in selected])),
                            accuracy_std=float(np.std([r['accuracy'] for r in selected],ddof=1)),
                            f1=float(np.mean([r['f1'] for r in selected])),runs=len(selected)))
    ranking.sort(key=lambda r:(-r['accuracy'],-r['f1'],r['name']))
    save_json(out/'cv_selection.json',dict(winner=ranking[0]['name'],ranking=ranking,
        config=next(c for c in options if c['name']==ranking[0]['name'])))
    assert preservation()==json.loads((out/'input_hashes.json').read_text())
    print('CV COMPLETE\n'+json.dumps(ranking,indent=2),flush=True)


def confirm(out):
    if (out/'confirmation_started.json').exists():
        raise FileExistsError('Confirmation already started')
    chosen=json.loads((out/'cv_selection.json').read_text())
    save_json(out/'confirmation_started.json',dict(cv_selection_sha256=digest(out/'cv_selection.json')))
    fitting=pd.read_csv(ROOT/'data/processed/train.csv')
    validation=pd.read_csv(ROOT/'data/processed/validation.csv')
    seeds=[42,123,2026,7,99]
    arrays=[]
    records=[]
    for seed in seeds:
        name=f'deploy_seed{seed}'
        model,vocab,record=train(chosen['config'],seed,fitting,validation,out,name)
        p=probabilities(model,encode(validation.text.tolist(),vocab,64))
        arrays.append(p)
        record['validation']=metrics(validation.target,p)
        records.append(record)
        np.save(out/f'{name}_validation.npy',p)
        print(f"{name}: {record['validation']}",flush=True)
    save_json(out/'confirmation_runs.json',records)
    candidates=[]
    for ensemble,p in [(False,arrays[0]),(True,np.mean(arrays,axis=0))]:
        for threshold in np.round(np.arange(.35,.651,.01),2):
            candidates.append(dict(ensemble=ensemble,threshold=float(threshold),**metrics(validation.target,p,threshold)))
    candidates.sort(key=lambda r:(-r['accuracy'],-r['f1'],abs(r['threshold']-.5),r['ensemble']))
    winner=candidates[0]
    deployed=seeds if winner['ensemble'] else [42]
    p=np.mean(arrays,axis=0) if winner['ensemble'] else arrays[0]
    previous=json.loads((ROOT/'experiments/mlp_improved_2026_09_18/selection.json').read_text())['validation_selected_threshold']
    gate=winner['accuracy']>=previous['accuracy'] and winner['f1']>=previous['f1']
    selection=dict(config=chosen['config'],seeds=deployed,ensemble=winner['ensemble'],threshold=winner['threshold'],
                   validation=winner,previous_validation=previous,validation_gate_passed=gate,
                   checkpoint_sha256={f'deploy_seed{seed}.pt':digest(out/f'deploy_seed{seed}.pt') for seed in deployed})
    pd.DataFrame(candidates).to_csv(out/'validation_thresholds.csv',index=False)
    rows=validation[['id','target']].copy()
    rows['probability']=p
    rows.to_csv(out/'validation_predictions.csv',index=False)
    assert preservation()==json.loads((out/'input_hashes.json').read_text())
    save_json(out/'selection.json',selection)
    print('FINAL SELECTION LOCKED\n'+json.dumps(selection,indent=2),flush=True)


def load(out):
    selection=json.loads((out/'selection.json').read_text())
    models=[]
    for seed in selection['seeds']:
        name=f'deploy_seed{seed}.pt'
        assert digest(out/name)==selection['checkpoint_sha256'][name]
        checkpoint=torch.load(out/name,map_location='cpu',weights_only=True)
        model=FinalMLP(checkpoint['config'],len(checkpoint['vocabulary']))
        model.load_state_dict(checkpoint['state_dict'])
        models.append((model.eval(),checkpoint['vocabulary']))
    return selection,models


def predict(loaded,texts):
    return np.mean([probabilities(model,encode(list(texts),vocab,64)) for model,vocab in loaded[1]],axis=0)


def evaluate(out):
    if (out/'evaluation_started.json').exists():
        raise FileExistsError('Evaluation already started; do not alter selection using test')
    loaded=load(out)
    selection=loaded[0]
    assert preservation()==json.loads((out/'input_hashes.json').read_text())
    save_json(out/'evaluation_started.json',dict(selection_sha256=digest(out/'selection.json'),time_unix=time.time()))
    test=pd.read_csv(ROOT/'data/processed/test.csv')
    p=predict(loaded,test.text.tolist())
    rows=test[['id','text','target']].copy()
    rows['probability']=p
    rows['prediction']=(p>=selection['threshold']).astype(int)
    rows.to_csv(out/'test_predictions.csv',index=False)
    previous=pd.read_csv(ROOT/'experiments/mlp_improved_2026_09_18/test_predictions.csv')
    assert np.array_equal(test.id,previous.id) and np.array_equal(test.target,previous.target)
    old=metrics(test.target,previous.probability,.49)
    new=metrics(test.target,p,selection['threshold'])
    comparison=[dict(model='September improved MLP',**old),dict(model='Final CV-selected MLP',**new)]
    rng=np.random.default_rng(20261016)
    deltas=[]
    y,bp=test.target.to_numpy(),previous.probability.to_numpy()
    for _ in range(2000):
        idx=rng.integers(0,len(y),len(y))
        a,b=metrics(y[idx],bp[idx],.49),metrics(y[idx],p[idx],selection['threshold'])
        deltas.append([b[k]-a[k] for k in ['accuracy','f1']])
    intervals={key:np.percentile(np.array(deltas)[:,i],[2.5,97.5]).tolist() for i,key in enumerate(['accuracy','f1'])}
    save_json(out/'summary.json',dict(comparison=comparison,improvement={k:new[k]-old[k] for k in new},
        paired_bootstrap_95_percent_delta_intervals=intervals,validation_gate_passed=selection['validation_gate_passed'],
        caveat='Existing reused local test. CV was used for model selection. Conditional tweet bootstrap omits seed/search uncertainty and event correlation.',
        previous_models_and_data_preserved=preservation()==json.loads((out/'input_hashes.json').read_text())))
    pd.DataFrame(comparison).to_csv(out/'comparison.csv',index=False)
    kaggle=pd.read_csv(ROOT/'data/raw/test.csv')
    pd.DataFrame(dict(id=kaggle.id,target=(predict(loaded,kaggle.text.tolist())>=selection['threshold']).astype(int))).to_csv(out/'kaggle_submission.csv',index=False)
    print(pd.DataFrame(comparison).to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','search','confirm','evaluate','predict'])
    parser.add_argument('--run-name',default=DEFAULT_RUN)
    parser.add_argument('--text')
    parser.add_argument('--source',choices=list(SOURCES))
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    out=ROOT/'experiments'/args.run_name
    if args.command=='prepare': prepare(args.source)
    elif args.command=='search': search(out)
    elif args.command=='confirm': confirm(out)
    elif args.command=='evaluate': evaluate(out)
    else:
        if args.text is None: parser.error('--text required')
        loaded=load(out)
        p=float(predict(loaded,[args.text])[0])
        print(json.dumps(dict(disaster_probability=p,threshold=loaded[0]['threshold'],prediction=int(p>=loaded[0]['threshold'])),indent=2))


if __name__=='__main__':
    main()
