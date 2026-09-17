"""Train-only feature fitting and validation-selected stronger MLP experiments.

Commands: prepare-embeddings, search, evaluate, predict. Search never reads test labels.
Evaluation is a separate step after selection.json has fixed the configuration.
"""
import argparse
import copy
import gzip
import hashlib
import json
import random
import time
from pathlib import Path

from project import ROOT, clean, encode, save_json
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch import nn

EMBEDDING_NAME = 'glove-wiki-gigaword-100'
EMBEDDING_MD5 = '40ec481866001177b8cd4cb0df92924f'
EMBEDDING_URL = 'https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz'
DEFAULT_RUN = 'mlp_improved_2026_09_18'
SEEDS = [42, 123, 2026]


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def metrics(labels, probabilities, threshold=.5):
    predicted = np.asarray(probabilities) >= threshold
    return dict(accuracy=float(accuracy_score(labels, predicted)),
                precision=float(precision_score(labels, predicted, zero_division=0)),
                recall=float(recall_score(labels, predicted, zero_division=0)),
                f1=float(f1_score(labels, predicted, zero_division=0)))


def prepare_embeddings():
    """Read the verified text vectors; retain only the training vocabulary."""
    folder = ROOT / 'data/embeddings'
    path = folder / f'{EMBEDDING_NAME}.gz'
    if not path.exists():
        partial = path.with_suffix('.gz.part')
        if not partial.exists():
            raise FileNotFoundError(f'Download {EMBEDDING_URL} to {partial} first.')
        if digest(partial, 'md5') != EMBEDDING_MD5:
            raise ValueError('Embedding download checksum mismatch')
        partial.rename(path)
    if digest(path, 'md5') != EMBEDDING_MD5:
        raise ValueError('Embedding checksum mismatch')
    vocab = json.loads((ROOT / 'data/processed/vocabulary.json').read_text())
    # Small deterministic random vectors for train-only words absent in GloVe.
    matrix = np.random.default_rng(42).normal(0, .05, (len(vocab), 100)).astype('float32')
    found = set()
    with gzip.open(path, 'rt', encoding='utf-8') as source:
        assert source.readline().strip() == '400000 100'
        for line in source:
            word, _, vector = line.rstrip().partition(' ')
            if word in vocab:
                values = np.fromstring(vector, sep=' ', dtype=np.float32)
                if len(values) != 100:
                    raise ValueError(f'Invalid vector for {word}')
                matrix[vocab[word]] = values
                found.add(word)
    matrix[0] = 0
    matrix[1] = np.mean(matrix[[vocab[w] for w in sorted(found)]], axis=0)
    np.save(folder / 'glove_train_vocabulary.npy', matrix)
    train = pd.read_csv(ROOT / 'data/processed/train.csv')
    tokens = ' '.join(train.text.map(clean)).split()
    save_json(folder / 'provenance.json', {
        'source': EMBEDDING_URL, 'original_source': 'https://nlp.stanford.edu/projects/glove/',
        'license': 'https://opendatacommons.org/licenses/pddl/1-0/',
        'description': 'GloVe 6B Wikipedia 2014 + Gigaword, 100 dimensions; Gensim text-format mirror',
        'expected_md5': EMBEDDING_MD5, 'sha256': digest(path),
        'training_vocabulary_sha256': digest(ROOT / 'data/processed/vocabulary.json'),
        'matrix_sha256': digest(folder / 'glove_train_vocabulary.npy'),
        'matched_training_words': len(found), 'training_vocabulary_words': len(vocab) - 2,
        'training_token_coverage': sum(w in found for w in tokens) / len(tokens),
        'unknown_policy': 'Missing train words: seeded N(0, .05); UNK: mean of matched vectors; PAD: zero.'})
    print(f'GloVe matched {len(found)}/{len(vocab)-2} training words; token coverage {sum(w in found for w in tokens)/len(tokens):.2%}', flush=True)


class GloveMLP(nn.Module):
    """Pretrained word embeddings -> masked pooling -> feed-forward classifier."""
    def __init__(self, config, vocabulary_size, weights=None):
        super().__init__()
        self.pool = config['pool']
        self.embedding = nn.Embedding(vocabulary_size, 100, padding_idx=0)
        if weights is not None:
            with torch.no_grad():
                self.embedding.weight.copy_(torch.as_tensor(weights))
        self.embedding.weight.requires_grad_(not config['frozen'])
        width = 200 if self.pool == 'mean_max' else 100
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Dropout(config['dropout']),
                                  nn.Linear(width, config['hidden']), nn.ReLU(),
                                  nn.Dropout(config['dropout']), nn.Linear(config['hidden'], 1))

    def forward(self, tokens):
        mask = tokens.ne(0).unsqueeze(-1)
        embedded = self.embedding(tokens)
        mean = (embedded * mask).sum(1) / mask.sum(1).clamp(min=1)
        if self.pool == 'mean_max':
            maximum = embedded.masked_fill(~mask, -torch.inf).amax(1)
            maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
            mean = torch.cat((mean, maximum), dim=1)
        return self.head(mean).squeeze(-1)


class TfidfMLP(nn.Module):
    """Training-fitted TF-IDF features -> nonlinear hidden layer -> one logit."""
    def __init__(self, config, input_size):
        super().__init__()
        self.head = nn.Sequential(nn.Dropout(config['dropout']), nn.Linear(input_size, config['hidden']),
                                  nn.ReLU(), nn.Dropout(config['dropout']), nn.Linear(config['hidden'], 1))

    def forward(self, features):
        return self.head(features).squeeze(-1)


def make_vectorizer(characters):
    word = TfidfVectorizer(preprocessor=clean, lowercase=False, ngram_range=(1, 2),
                           min_df=2, max_df=.98, max_features=16000, sublinear_tf=True,
                           dtype=np.float32, token_pattern=r"(?u)\b\w[\w']*\b")
    if not characters:
        return word
    char = TfidfVectorizer(preprocessor=clean, lowercase=False, analyzer='char_wb',
                           ngram_range=(3, 5), min_df=3, max_features=12000,
                           sublinear_tf=True, dtype=np.float32)
    return FeatureUnion([('words', word), ('characters', char)], transformer_weights={'words': 1., 'characters': .5})


def as_batch(features, indices):
    if sparse.issparse(features):
        return torch.from_numpy(features[indices].toarray())
    return features[indices]


def predict_probabilities(model, features):
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, features.shape[0], 256):
            output.append(model(as_batch(features, np.arange(start, min(start + 256, features.shape[0])))).sigmoid().numpy())
    return np.concatenate(output)


def configurations():
    configs = []
    for frozen in [True, False]:
        for pool in ['mean', 'mean_max']:
            for hidden, dropout in [(64, .3), (128, .5)]:
                for weighted in [False, True]:
                    configs.append(dict(family='glove', frozen=frozen, pool=pool, hidden=hidden,
                                        dropout=dropout, weighted=weighted, lr=.001, embedding_lr=.0002,
                                        weight_decay=.01))
    for chars in [False, True]:
        for hidden in [64, 128]:
            for decay in [.01, .1]:
                configs.append(dict(family='tfidf', characters=chars, hidden=hidden, dropout=.4,
                                    weighted=False, lr=.001, weight_decay=decay))
    return [dict(c, trial=f'trial_{i:02}', epochs=40, patience=7, batch_size=64)
            for i, c in enumerate(configs, 1)]


def feature_key(config):
    return 'glove' if config['family'] == 'glove' else f"tfidf_{int(config['characters'])}"


def fit(config, seed, features, labels, out, weights, vocabulary):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    start = time.perf_counter()
    if config['family'] == 'glove':
        model = GloveMLP(config, len(vocabulary), weights)
        groups = [{'params': model.head.parameters(), 'lr': config['lr']}]
        if not config['frozen']:
            groups.append({'params': model.embedding.parameters(), 'lr': config['embedding_lr']})
    else:
        model = TfidfMLP(config, features['train'].shape[1])
        groups = [{'params': model.parameters(), 'lr': config['lr']}]
    optimizer = torch.optim.AdamW(groups, weight_decay=config['weight_decay'])
    positive_weight = float((labels['train'] == 0).sum() / (labels['train'] == 1).sum()) if config['weighted'] else 1.
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight))
    rng = np.random.default_rng(seed)
    history = []
    best_key = (-1., -1.)
    stale = 0
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        order = rng.permutation(len(labels['train']))
        loss_sum = 0.
        for begin in range(0, len(order), config['batch_size']):
            idx = order[begin:begin + config['batch_size']]
            optimizer.zero_grad()
            loss = loss_function(model(as_batch(features['train'], idx)), torch.from_numpy(labels['train'][idx]))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(idx)
        val_prob = predict_probabilities(model, features['validation'])
        result = metrics(labels['validation'], val_prob)
        history.append(dict(epoch=epoch, train_loss=loss_sum / len(order), **result))
        key = (result['f1'], result['accuracy'])
        if key > best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= config['patience']:
            break
    model.load_state_dict(best_state)
    validation_prob = predict_probabilities(model, features['validation'])
    name = f"{config['trial']}_seed{seed}"
    torch.save(dict(config=config, seed=seed, state_dict=best_state,
                    vocabulary=vocabulary if config['family'] == 'glove' else None,
                    input_size=features['train'].shape[1], max_length=64), out / f'{name}.pt')
    np.save(out / f'{name}_validation.npy', validation_prob)
    pd.DataFrame(history).to_csv(out / f'{name}_history.csv', index=False)
    record = dict(trial=config['trial'], seed=seed, config=config, best_epoch=best_epoch,
                  epochs_run=len(history), seconds=time.perf_counter() - start,
                  parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  validation=metrics(labels['validation'], validation_prob))
    save_json(out / f'{name}.json', record)
    print(f"{name}: val accuracy={record['validation']['accuracy']:.4f}, F1={record['validation']['f1']:.4f}, epoch={best_epoch}, seconds={record['seconds']:.1f}", flush=True)
    return record


def input_hashes():
    names = ['data/processed/train.csv', 'data/processed/validation.csv', 'data/processed/test.csv',
             'data/processed/vocabulary.json', 'artifacts/mlp.pt', 'reports/results.json']
    prior_checkpoint = 'experiments/mlp_tuning_2026_09_17/mlp_tuned.pt'
    if (ROOT / prior_checkpoint).exists():
        names.append(prior_checkpoint)
    return {name: digest(ROOT / name) for name in names}


def search(out):
    out.mkdir(parents=True, exist_ok=False)
    configs = configurations()
    save_json(out / 'search_plan.json', dict(configurations=configs, seeds=SEEDS,
        screening='24 configurations at seed 42; top three by validation F1, then accuracy at threshold 0.5.',
        confirmation='Top three configurations run at seeds 123 and 2026. Select highest mean seed validation F1; tie by mean accuracy.',
        deployment='For winning configuration, compare designated seed 42 and equal-weight three-seed probability ensemble by validation F1 at 0.5. Prefer single seed on exact tie.',
        threshold='Select on validation F1 from 0.30..0.65 in steps of .01; break ties by accuracy, then proximity to .5.',
        evaluation='Test labels are not loaded during search. Separate evaluate command after selection.json is saved. Existing test has been used in prior project iterations; comparison is exploratory.',
        objective='Disaster-class F1 is primary; report accuracy, precision and recall as well. No test-guided search.',
        pretrained='GloVe 6B 100d; external unsupervised pretraining, supervised vocabulary/TF-IDF fit on training split only.'))
    save_json(out / 'input_hashes.json', input_hashes())
    frames = {split: pd.read_csv(ROOT / f'data/processed/{split}.csv') for split in ['train', 'validation']}
    labels = {split: frame.target.to_numpy(dtype=np.float32) for split, frame in frames.items()}
    vocabulary = json.loads((ROOT / 'data/processed/vocabulary.json').read_text())
    weights = np.load(ROOT / 'data/embeddings/glove_train_vocabulary.npy')
    provenance = json.loads((ROOT / 'data/embeddings/provenance.json').read_text())
    assert provenance['training_vocabulary_sha256'] == digest(ROOT / 'data/processed/vocabulary.json')
    assert provenance['matrix_sha256'] == digest(ROOT / 'data/embeddings/glove_train_vocabulary.npy')
    save_json(out / 'embedding_provenance.json', provenance)
    features = {'glove': {split: encode(frame.text.tolist(), vocabulary, 64) for split, frame in frames.items()}}
    for chars in [False, True]:
        name = f'tfidf_{int(chars)}'
        vectorizer = make_vectorizer(chars)
        features[name] = dict(train=vectorizer.fit_transform(frames['train'].text).tocsr(),
                              validation=vectorizer.transform(frames['validation'].text).tocsr())
        joblib.dump(vectorizer, out / f'{name}_vectorizer.joblib')
    records = []
    for config in configs:
        records.append(fit(config, 42, features[feature_key(config)], labels, out, weights, vocabulary))
        save_json(out / 'all_runs.json', records)
    top = sorted(records, key=lambda r: (-r['validation']['f1'], -r['validation']['accuracy'], r['trial']))[:3]
    for row in top:
        for seed in SEEDS[1:]:
            records.append(fit(row['config'], seed, features[feature_key(row['config'])], labels, out, weights, vocabulary))
            save_json(out / 'all_runs.json', records)
    rankings = []
    for row in top:
        selected = [r for r in records if r['trial'] == row['trial']]
        rankings.append(dict(trial=row['trial'], mean_f1=float(np.mean([r['validation']['f1'] for r in selected])),
                             std_f1=float(np.std([r['validation']['f1'] for r in selected], ddof=1)),
                             mean_accuracy=float(np.mean([r['validation']['accuracy'] for r in selected]))))
    rankings.sort(key=lambda r: (-r['mean_f1'], -r['mean_accuracy'], r['trial']))
    winner = rankings[0]['trial']
    config = next(r['config'] for r in top if r['trial'] == winner)
    arrays = [np.load(out / f'{winner}_seed{seed}_validation.npy') for seed in SEEDS]
    single = metrics(labels['validation'], arrays[0])
    ensemble_prob = np.mean(arrays, axis=0)
    ensemble = metrics(labels['validation'], ensemble_prob)
    use_ensemble = (ensemble['f1'], ensemble['accuracy']) > (single['f1'], single['accuracy'])
    probabilities = ensemble_prob if use_ensemble else arrays[0]
    thresholds = [dict(threshold=float(t), **metrics(labels['validation'], probabilities, t))
                  for t in np.round(np.arange(.30, .651, .01), 2)]
    thresholds.sort(key=lambda r: (-r['f1'], -r['accuracy'], abs(r['threshold'] - .5)))
    threshold = thresholds[0]['threshold']
    selection = dict(winner=winner, config=config, rankings=rankings,
                     seeds=SEEDS if use_ensemble else [42], ensemble=use_ensemble,
                     single_seed_validation=single, ensemble_validation=ensemble,
                     threshold=threshold, validation_selected_threshold=thresholds[0])
    pd.DataFrame(thresholds).sort_values('threshold').to_csv(out / 'thresholds.csv', index=False)
    np.save(out / 'selected_validation_probabilities.npy', probabilities)
    errors = frames['validation'][['id', 'text', 'target']].copy()
    errors['probability'] = probabilities
    errors['prediction'] = (probabilities >= threshold).astype(int)
    errors.loc[errors.prediction != errors.target].to_csv(out / 'validation_errors.csv', index=False)
    assert input_hashes() == json.loads((out / 'input_hashes.json').read_text())
    selection['checkpoint_sha256'] = {f'{winner}_seed{seed}.pt': digest(out / f'{winner}_seed{seed}.pt') for seed in selection['seeds']}
    if config['family'] == 'tfidf':
        name = f'{feature_key(config)}_vectorizer.joblib'
        selection['vectorizer_sha256'] = {name: digest(out / name)}
    save_json(out / 'selection.json', selection)
    print('SELECTION FIXED; test labels not read.\n' + json.dumps(selection, indent=2), flush=True)


def load_selected(out):
    selection = json.loads((out / 'selection.json').read_text())
    for filename, checksum in {**selection['checkpoint_sha256'], **selection.get('vectorizer_sha256', {})}.items():
        if digest(out / filename) != checksum:
            raise ValueError(f'Selected artifact changed: {filename}')
    models = []
    vocabulary = None
    for seed in selection['seeds']:
        checkpoint = torch.load(out / f"{selection['winner']}_seed{seed}.pt", map_location='cpu', weights_only=True)
        config = checkpoint['config']
        assert config == selection['config']
        if config['family'] == 'glove':
            vocabulary = checkpoint['vocabulary']
            model = GloveMLP(config, len(vocabulary))
        else:
            model = TfidfMLP(config, checkpoint['input_size'])
        model.load_state_dict(checkpoint['state_dict'])
        models.append(model.eval())
    # Only load locally produced trusted sklearn serialization.
    vectorizer = joblib.load(out / f"{feature_key(selection['config'])}_vectorizer.joblib") if selection['config']['family'] == 'tfidf' else None
    return selection, models, vocabulary, vectorizer


def selected_probabilities(loaded, texts):
    selection, models, vocabulary, vectorizer = loaded
    features = encode(list(texts), vocabulary, 64) if vocabulary is not None else vectorizer.transform(texts).tocsr()
    return np.mean([predict_probabilities(model, features) for model in models], axis=0)


def evaluate(out):
    if (out / 'evaluation_started.json').exists():
        raise FileExistsError('Evaluation has already been started; review existing artifacts, do not retune to this test.')
    loaded = load_selected(out)
    selection = loaded[0]
    assert input_hashes() == json.loads((out / 'input_hashes.json').read_text())
    save_json(out / 'evaluation_started.json', {'selection_sha256': digest(out / 'selection.json'), 'time_unix': time.time()})
    # Test labels first loaded only after the selected model and threshold are fixed.
    test = pd.read_csv(ROOT / 'data/processed/test.csv')
    probabilities = selected_probabilities(loaded, test.text.tolist())
    rows = test[['id', 'text', 'target']].copy()
    rows['probability'] = probabilities
    rows['prediction'] = (probabilities >= selection['threshold']).astype(int)
    rows.to_csv(out / 'test_predictions.csv', index=False)
    old = pd.read_csv(ROOT / 'reports/mlp_test_predictions.csv')
    previous = pd.read_csv(ROOT / 'experiments/mlp_tuning_2026_09_17/tuned_mlp_validation_threshold_test_predictions.csv')
    assert np.array_equal(old.id, test.id) and np.array_equal(previous.id, test.id)
    baseline = metrics(test.target, old.probability)
    prior = metrics(test.target, previous.probability, .48)
    current = metrics(test.target, probabilities, selection['threshold'])
    comparison = [dict(model='Original MLP', threshold=.5, **baseline),
                  dict(model='September 17 tuned MLP', threshold=.48, **prior),
                  dict(model='Improved MLP (selected)', threshold=selection['threshold'], **current),
                  dict(model='Improved MLP (fixed 0.5)', threshold=.5, **metrics(test.target, probabilities))]
    pd.DataFrame(comparison).to_csv(out / 'comparison.csv', index=False)
    rng = np.random.default_rng(20260918)
    bootstrap = []
    y = test.target.to_numpy()
    bp = old.probability.to_numpy()
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        a, b = metrics(y[idx], bp[idx]), metrics(y[idx], probabilities[idx], selection['threshold'])
        bootstrap.append([b[k] - a[k] for k in ['accuracy', 'f1']])
    confidence = {k: np.percentile(np.array(bootstrap)[:, i], [2.5, 97.5]).tolist() for i, k in enumerate(['accuracy', 'f1'])}
    save_json(out / 'summary.json', dict(comparison=comparison,
        improvement_vs_original={k: current[k] - baseline[k] for k in current},
        improvement_vs_previous={k: current[k] - prior[k] for k in current},
        paired_bootstrap_95_percent_delta_intervals=confidence,
        uncertainty='Conditional tweet-level paired bootstrap; excludes seed/search uncertainty and event correlation. Existing reused test set, so improvement is exploratory.',
        input_hashes_unchanged=input_hashes() == json.loads((out / 'input_hashes.json').read_text())))
    competition = pd.read_csv(ROOT / 'data/raw/test.csv')
    cp = selected_probabilities(loaded, competition.text.tolist())
    pd.DataFrame({'id': competition.id, 'target': (cp >= selection['threshold']).astype(int)}).to_csv(out / 'kaggle_submission.csv', index=False)
    print(pd.DataFrame(comparison).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare-embeddings', 'search', 'evaluate', 'predict'])
    parser.add_argument('--run-name', default=DEFAULT_RUN)
    parser.add_argument('--text')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    out = ROOT / 'experiments' / args.run_name
    if args.command == 'prepare-embeddings':
        prepare_embeddings()
    elif args.command == 'search':
        search(out)
    elif args.command == 'evaluate':
        evaluate(out)
    else:
        if args.text is None:
            parser.error('--text is required for predict')
        loaded = load_selected(out)
        p = float(selected_probabilities(loaded, [args.text])[0])
        threshold = loaded[0]['threshold']
        print(json.dumps(dict(disaster_probability=p, threshold=threshold, prediction=int(p >= threshold),
                              label='disaster' if p >= threshold else 'non-disaster'), indent=2))


if __name__ == '__main__':
    main()
