"""Validation-only accuracy refinement of the three confirmed MLP finalists.

Run after improve_mlp.py search and before evaluate. Preserve the F1-first selection.
"""
import argparse
import json
from improve_mlp import ROOT, DEFAULT_RUN, SEEDS, digest, metrics, save_json
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default=DEFAULT_RUN)
    args = parser.parse_args()
    out = ROOT / 'experiments' / args.run_name
    if (out / 'evaluation_started.json').exists():
        raise RuntimeError('Cannot refine selection after final test evaluation')
    if (out / 'accuracy_refinement_plan.json').exists():
        raise RuntimeError('Refinement already started; preserve the existing result')
    original = json.loads((out / 'selection.json').read_text())
    save_json(out / 'selection_f1_screening.json', original)
    save_json(out / 'accuracy_refinement_plan.json', {
        'reason': 'User prioritized accuracy after small gains in the preceding experiment. Top confirmed configurations have nearly identical mean F1 but different accuracy.',
        'candidate_configs': [r['trial'] for r in original['rankings']],
        'deployment': 'Compare designated seed 42 and equal-weight three-seed ensemble for each of three confirmed finalists.',
        'threshold_grid': np.round(np.arange(.30, .651, .01), 2).tolist(),
        'objective': 'Maximize validation accuracy; break ties by disaster F1, proximity to threshold 0.5, single model preferred, trial id.',
        'test_policy': 'Refinement is adaptive to validation only, before any evaluation of new models on test. Preserve original F1-first selection and record this change in objective.'})
    validation = pd.read_csv(ROOT / 'data/processed/validation.csv')
    candidates = []
    for rank in original['rankings']:
        trial = rank['trial']
        arrays = [np.load(out / f'{trial}_seed{seed}_validation.npy') for seed in SEEDS]
        for ensemble, probabilities in [(False, arrays[0]), (True, np.mean(arrays, axis=0))]:
            for threshold in np.round(np.arange(.30, .651, .01), 2):
                candidates.append(dict(trial=trial, ensemble=ensemble, threshold=float(threshold),
                                       **metrics(validation.target, probabilities, threshold)))
    candidates.sort(key=lambda r: (-r['accuracy'], -r['f1'], abs(r['threshold'] - .5), r['ensemble'], r['trial']))
    winner = candidates[0]
    records = json.loads((out / 'all_runs.json').read_text())
    config = next(r['config'] for r in records if r['trial'] == winner['trial'])
    seeds = SEEDS if winner['ensemble'] else [42]
    arrays = [np.load(out / f"{winner['trial']}_seed{seed}_validation.npy") for seed in SEEDS]
    probabilities = np.mean(arrays, axis=0) if winner['ensemble'] else arrays[0]
    selection = dict(winner=winner['trial'], config=config, rankings=original['rankings'],
                     seeds=seeds, ensemble=winner['ensemble'], threshold=winner['threshold'],
                     selection_objective='Validation accuracy first, disaster F1 as tie-breaker; see accuracy_refinement_plan.json',
                     single_seed_validation=metrics(validation.target, arrays[0]),
                     ensemble_validation=metrics(validation.target, np.mean(arrays, axis=0)),
                     validation_selected_threshold={k: winner[k] for k in ['threshold', 'accuracy', 'precision', 'recall', 'f1']},
                     checkpoint_sha256={f"{winner['trial']}_seed{seed}.pt": digest(out / f"{winner['trial']}_seed{seed}.pt") for seed in seeds})
    if config['family'] == 'tfidf':
        name = f"tfidf_{int(config['characters'])}_vectorizer.joblib"
        selection['vectorizer_sha256'] = {name: digest(out / name)}
    pd.DataFrame(candidates).to_csv(out / 'accuracy_refinement_candidates.csv', index=False)
    np.save(out / 'selected_validation_probabilities.npy', probabilities)
    errors = validation[['id', 'text', 'target']].copy()
    errors['probability'] = probabilities
    errors['prediction'] = (probabilities >= winner['threshold']).astype(int)
    errors.loc[errors.prediction != errors.target].to_csv(out / 'validation_errors.csv', index=False)
    save_json(out / 'selection.json', selection)
    print(json.dumps(selection, indent=2))


if __name__ == '__main__':
    main()
