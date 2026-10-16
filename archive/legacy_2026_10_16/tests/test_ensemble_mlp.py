import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from ensemble_mlp import ROOT,DEFAULT_RUN,predict,source_hashes
from improve_mlp import metrics


class MLPBlendTests(unittest.TestCase):
    def test_selection_is_a_valid_probability_mixture(self):
        out=ROOT/'experiments'/DEFAULT_RUN
        if not (out/'selection.json').exists():self.skipTest('Selection not run')
        selected=json.loads((out/'selection.json').read_text())
        self.assertAlmostEqual(sum(selected['weights']),1.)
        self.assertTrue(all(0<=w<=1 for w in selected['weights']))
        self.assertEqual(source_hashes(),selected['source_selection_sha256'])
        p=np.array(selected['weights'])@np.load(out/'member_validation_probabilities.npy')
        saved=pd.read_csv(out/'validation_predictions.csv')
        np.testing.assert_allclose(p,saved.probability,atol=1e-7)
        for k,v in metrics(saved.target,p,selected['threshold']).items(): self.assertAlmostEqual(v,selected[k])

    def test_saved_test_predictions_reproduce(self):
        out=ROOT/'experiments'/DEFAULT_RUN
        if not (out/'summary.json').exists(): self.skipTest('Evaluation not run')
        torch.set_num_threads(4)
        rows=pd.read_csv(out/'test_predictions.csv')
        selected=json.loads((out/'selection.json').read_text())
        p=predict(out,rows.text.tolist())
        np.testing.assert_allclose(p,rows.probability,atol=1e-6)
        expected=json.loads((out/'summary.json').read_text())['comparison'][1]
        for k,v in metrics(rows.target,p,selected['threshold']).items(): self.assertAlmostEqual(v,expected[k])


if __name__=='__main__': unittest.main()
