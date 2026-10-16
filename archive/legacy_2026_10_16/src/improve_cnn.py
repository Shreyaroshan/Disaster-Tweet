"""Pretrained CNN tuning, with a fixed validation-only accuracy selection protocol."""
import argparse
import copy
import json
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from project import ROOT, encode, save_json
from improve_mlp import digest, metrics

DEFAULT_RUN = 'cnn_improved_2026_09_18'
SEEDS = [42, 123, 2026]


class PretrainedCNN(nn.Module):
    def __init__(self, config, vocabulary_size, weights=None):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(vocabulary_size, 100, padding_idx=0)
        if weights is not None:
            with torch.no_grad():
                self.embedding.weight.copy_(torch.as_tensor(weights))
        self.embedding.weight.requires_grad_(not config['frozen'])
        self.embedding_dropout = nn.Dropout1d(.1)
        self.convs = nn.ModuleList([nn.Conv1d(100, config['filters'], k) for k in config['kernels']])
        width = len(config['kernels']) * config['filters'] * (2 if config['pool'] == 'max_mean' else 1)
        self.head = nn.Sequential(nn.Dropout(config['dropout']), nn.Linear(width, 1))

    def forward(self, tokens):
        lengths = tokens.ne(0).sum(1)
        # All encodings use right padding. Trim unused positions without truncating words.
        width = max(max(self.config['kernels']), int(lengths.max()))
        tokens = tokens[:, :width]
        if tokens.shape[1] < width:
            tokens = nn.functional.pad(tokens, (0, width - tokens.shape[1]))
        embedded = self.embedding_dropout(self.embedding(tokens).transpose(1, 2))
        pooled = []
        for kernel, conv in zip(self.config['kernels'], self.convs):
            features = torch.relu(conv(embedded))
            valid = (lengths - kernel + 1).clamp(min=1)
            mask = torch.arange(features.shape[2])[None, :] < valid[:, None]
            pooled.append(features.masked_fill(~mask[:, None, :], -torch.inf).amax(2))
            if self.config['pool'] == 'max_mean':
                pooled.append((features * mask[:, None, :]).sum(2) / valid[:, None])
        return self.head(torch.cat(pooled, 1)).squeeze(-1)


def probabilities(model, tokens):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(batch).sigmoid() for batch in tokens.split(128)]).numpy()


def configurations():
    base = dict(frozen=False, filters=64, kernels=[2, 3, 4], pool='max', dropout=.5,
                lr=.001, embedding_lr=.0002, weight_decay=.01, epochs=30, patience=6, batch_size=64)
    changes = [dict(frozen=frozen, kernels=kernels) for frozen in [True, False]
               for kernels in [[2, 3, 4], [3, 4, 5]]]
    changes += [dict(filters=filters, dropout=dropout) for filters in [32, 96] for dropout in [.3, .5]]
    changes += [dict(pool='max_mean', dropout=dropout) for dropout in [.3, .5]]
    changes += [dict(embedding_lr=rate) for rate in [.00005, .001]]
    return [dict(base, **change, trial=f'trial_{i:02}') for i, change in enumerate(changes, 1)]


def preserved_hashes():
    names = ['data/processed/train.csv', 'data/processed/validation.csv', 'data/processed/test.csv',
             'data/processed/vocabulary.json', 'artifacts/cnn.pt', 'artifacts/mlp.pt', 'reports/results.json',
             'experiments/mlp_improved_2026_09_18/selection.json']
    return {name: digest(ROOT / name) for name in names if (ROOT / name).exists()}


def train(config, seed, x, y, vocabulary, weights, out):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = PretrainedCNN(config, len(vocabulary), weights)
    groups = [dict(params=list(model.convs.parameters()) + list(model.head.parameters()), lr=config['lr'])]
    if not config['frozen']:
        groups.append(dict(params=model.embedding.parameters(), lr=config['embedding_lr']))
    optimizer = torch.optim.AdamW(groups, weight_decay=config['weight_decay'])
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    history, best_key, stale = [], (-1., -1.), 0
    start = time.perf_counter()
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        order = rng.permutation(len(y['train']))
        total = 0.
        for begin in range(0, len(order), config['batch_size']):
            idx = order[begin:begin + config['batch_size']]
            optimizer.zero_grad()
            loss = loss_fn(model(x['train'][idx]), y['train'][idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        result = metrics(y['validation'].numpy(), probabilities(model, x['validation']))
        history.append(dict(epoch=epoch, train_loss=total/len(order), **result))
        key = (result['accuracy'], result['f1'])
        if key > best_key:
            best_key, best_state, best_epoch, stale = key, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= config['patience']:
            break
    model.load_state_dict(best_state)
    val = probabilities(model, x['validation'])
    name = f"{config['trial']}_seed{seed}"
    torch.save(dict(config=config, seed=seed, vocabulary=vocabulary, state_dict=best_state), out/f'{name}.pt')
    np.save(out/f'{name}_validation.npy', val)
    pd.DataFrame(history).to_csv(out/f'{name}_history.csv', index=False)
    result = dict(trial=config['trial'], seed=seed, config=config, best_epoch=best_epoch,
                  epochs_run=len(history), seconds=time.perf_counter()-start,
                  parameters=sum(p.numel() for p in model.parameters()), validation=metrics(y['validation'].numpy(), val))
    save_json(out/f'{name}.json', result)
    print(f"{name}: accuracy={result['validation']['accuracy']:.4f}, F1={result['validation']['f1']:.4f}, epoch={best_epoch}, seconds={result['seconds']:.1f}", flush=True)
    return result


def search(out):
    out.mkdir(parents=True, exist_ok=False)
    configs = configurations()
    save_json(out/'search_plan.json', dict(configurations=configs, seeds=SEEDS,
        objective='Validation accuracy first, disaster-class F1 breaks ties.',
        screening='12 configurations at seed 42; checkpoint by validation accuracy/F1 at threshold 0.5.',
        confirmation='Top three by validation accuracy/F1 repeated at seeds 123 and 2026; configuration chosen by mean seed accuracy, then mean F1.',
        deployment='For winning configuration, compare designated seed 42 and equal-weight three-seed probability ensemble over thresholds 0.30..0.65 (step .01). Select validation accuracy, then F1, proximity to .5, then single model.',
        test_policy='Separate evaluate command after selection and checkpoint hashes are saved; do not change selection using test results. Previously reused local test: exploratory comparison.'))
    save_json(out/'input_hashes.json', preserved_hashes())
    vocabulary = json.loads((ROOT/'data/processed/vocabulary.json').read_text())
    provenance = json.loads((ROOT/'data/embeddings/provenance.json').read_text())
    assert provenance['training_vocabulary_sha256'] == digest(ROOT/'data/processed/vocabulary.json')
    assert provenance['matrix_sha256'] == digest(ROOT/'data/embeddings/glove_train_vocabulary.npy')
    save_json(out/'embedding_provenance.json', provenance)
    weights = np.load(ROOT/'data/embeddings/glove_train_vocabulary.npy')
    frames = {split: pd.read_csv(ROOT/f'data/processed/{split}.csv') for split in ['train', 'validation']}
    x = {split: encode(frame.text.tolist(), vocabulary, 64) for split, frame in frames.items()}
    y = {split: torch.tensor(frame.target.to_numpy(), dtype=torch.float32) for split, frame in frames.items()}
    runs = []
    for config in configs:
        runs.append(train(config, 42, x, y, vocabulary, weights, out))
        save_json(out/'all_runs.json', runs)
    top = sorted(runs, key=lambda r: (-r['validation']['accuracy'], -r['validation']['f1'], r['trial']))[:3]
    for row in top:
        for seed in SEEDS[1:]:
            runs.append(train(row['config'], seed, x, y, vocabulary, weights, out))
            save_json(out/'all_runs.json', runs)
    ranking = []
    for row in top:
        values = [r['validation'] for r in runs if r['trial'] == row['trial']]
        ranking.append(dict(trial=row['trial'], mean_accuracy=float(np.mean([v['accuracy'] for v in values])),
                            std_accuracy=float(np.std([v['accuracy'] for v in values], ddof=1)), mean_f1=float(np.mean([v['f1'] for v in values]))))
    ranking.sort(key=lambda r: (-r['mean_accuracy'], -r['mean_f1'], r['trial']))
    winner = ranking[0]['trial']
    config = next(r['config'] for r in top if r['trial'] == winner)
    arrays = [np.load(out/f'{winner}_seed{seed}_validation.npy') for seed in SEEDS]
    candidates = []
    for ensemble, p in [(False, arrays[0]), (True, np.mean(arrays, axis=0))]:
        for threshold in np.round(np.arange(.30, .651, .01), 2):
            candidates.append(dict(ensemble=ensemble, threshold=float(threshold), **metrics(y['validation'].numpy(), p, threshold)))
    candidates.sort(key=lambda r: (-r['accuracy'], -r['f1'], abs(r['threshold']-.5), r['ensemble']))
    selected = candidates[0]
    seeds = SEEDS if selected['ensemble'] else [42]
    p = np.mean(arrays, axis=0) if selected['ensemble'] else arrays[0]
    selection = dict(winner=winner, config=config, rankings=ranking, seeds=seeds,
                     ensemble=selected['ensemble'], threshold=selected['threshold'], validation=selected,
                     checkpoint_sha256={f'{winner}_seed{seed}.pt': digest(out/f'{winner}_seed{seed}.pt') for seed in seeds})
    pd.DataFrame(candidates).to_csv(out/'threshold_candidates.csv', index=False)
    np.save(out/'selected_validation_probabilities.npy', p)
    errors = frames['validation'][['id', 'text', 'target']].copy()
    errors['probability'], errors['prediction'] = p, (p >= selected['threshold']).astype(int)
    errors.loc[errors.prediction != errors.target].to_csv(out/'validation_errors.csv', index=False)
    assert preserved_hashes() == json.loads((out/'input_hashes.json').read_text())
    save_json(out/'selection.json', selection)
    print('SELECTION FIXED; no test labels read.\n'+json.dumps(selection, indent=2), flush=True)


def load_selected(out):
    selection = json.loads((out/'selection.json').read_text())
    models = []
    for seed in selection['seeds']:
        filename = f"{selection['winner']}_seed{seed}.pt"
        assert digest(out/filename) == selection['checkpoint_sha256'][filename]
        cp = torch.load(out/filename, map_location='cpu', weights_only=True)
        assert cp['config'] == selection['config']
        model = PretrainedCNN(cp['config'], len(cp['vocabulary']))
        model.load_state_dict(cp['state_dict'])
        models.append(model.eval())
    return selection, models, cp['vocabulary']


def predict_selected(loaded, texts):
    selection, models, vocabulary = loaded
    x = encode(list(texts), vocabulary, 64)
    return np.mean([probabilities(model, x) for model in models], axis=0)


def evaluate(out):
    if (out/'evaluation_started.json').exists():
        raise FileExistsError('Final evaluation already started; retain existing results.')
    loaded = load_selected(out)
    selection = loaded[0]
    assert preserved_hashes() == json.loads((out/'input_hashes.json').read_text())
    save_json(out/'evaluation_started.json', dict(selection_sha256=digest(out/'selection.json'), time_unix=time.time()))
    test = pd.read_csv(ROOT/'data/processed/test.csv')
    p = predict_selected(loaded, test.text.tolist())
    baseline = pd.read_csv(ROOT/'reports/cnn_test_predictions.csv')
    assert np.array_equal(test.id, baseline.id) and np.array_equal(test.target, baseline.target)
    rows = test[['id', 'text', 'target']].copy()
    rows['probability'], rows['prediction'] = p, (p >= selection['threshold']).astype(int)
    rows.to_csv(out/'test_predictions.csv', index=False)
    comparison = [dict(model='Original CNN', threshold=.5, **metrics(test.target, baseline.probability)),
                  dict(model='Improved CNN', threshold=selection['threshold'], **metrics(test.target, p, selection['threshold'])),
                  dict(model='Improved CNN at 0.5', threshold=.5, **metrics(test.target, p))]
    mlp_path = ROOT/'experiments/mlp_improved_2026_09_18/test_predictions.csv'
    if mlp_path.exists():
        mlp = pd.read_csv(mlp_path)
        assert np.array_equal(test.id, mlp.id)
        comparison.append(dict(model='Improved MLP (reference)', threshold=.49, **metrics(test.target, mlp.probability, .49)))
    pd.DataFrame(comparison).to_csv(out/'comparison.csv', index=False)
    rng = np.random.default_rng(20260918)
    y, bp = test.target.to_numpy(), baseline.probability.to_numpy()
    deltas = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        a, b = metrics(y[idx], bp[idx]), metrics(y[idx], p[idx], selection['threshold'])
        deltas.append([b[k]-a[k] for k in ['accuracy', 'f1']])
    intervals = {key: np.percentile(np.array(deltas)[:, i], [2.5, 97.5]).tolist() for i, key in enumerate(['accuracy', 'f1'])}
    save_json(out/'summary.json', dict(comparison=comparison,
        improvement={k: comparison[1][k]-comparison[0][k] for k in ['accuracy','precision','recall','f1']},
        paired_bootstrap_95_percent_delta_intervals=intervals,
        caveat='Conditional tweet bootstrap excludes training/search variation and event correlation; reused local test set, exploratory comparison.',
        baseline_preserved=preserved_hashes() == json.loads((out/'input_hashes.json').read_text())))
    kaggle = pd.read_csv(ROOT/'data/raw/test.csv')
    kp = predict_selected(loaded, kaggle.text.tolist())
    pd.DataFrame(dict(id=kaggle.id, target=(kp >= selection['threshold']).astype(int))).to_csv(out/'kaggle_submission.csv', index=False)
    print(pd.DataFrame(comparison).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['search', 'evaluate', 'predict'])
    parser.add_argument('--run-name', default=DEFAULT_RUN)
    parser.add_argument('--text')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    out = ROOT/'experiments'/args.run_name
    if args.command == 'search':
        search(out)
    elif args.command == 'evaluate':
        evaluate(out)
    else:
        if args.text is None:
            parser.error('--text is required')
        loaded = load_selected(out)
        p = float(predict_selected(loaded, [args.text])[0])
        threshold = loaded[0]['threshold']
        print(json.dumps(dict(disaster_probability=p, threshold=threshold, prediction=int(p>=threshold)), indent=2))


if __name__ == '__main__':
    main()
