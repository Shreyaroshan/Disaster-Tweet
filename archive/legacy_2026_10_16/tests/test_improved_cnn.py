"""Padding, embedding-training and saved-model checks for the improved CNN."""
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from project import ROOT, encode
from improve_cnn import (PretrainedCNN, configurations, DEFAULT_RUN, load_selected,
                         predict_selected, preserved_hashes, metrics)


class ImprovedCNNTests(unittest.TestCase):
    def test_short_empty_and_padding_invariance(self):
        torch.set_num_threads(4)
        vocab = {'<PAD>':0, '<UNK>':1, 'fire':2, 'flood':3}
        for pooling in ['max', 'max_mean']:
            config = dict(configurations()[0], pool=pooling)
            model = PretrainedCNN(config, len(vocab)).eval()
            for text in ['', 'fire', 'fire flood fire flood fire']:
                x = encode([text], vocab, 10)
                torch.testing.assert_close(model(x), model(torch.nn.functional.pad(x, (0, 20))))
                self.assertTrue(torch.isfinite(model(x)).all())
            self.assertTrue(torch.isfinite(model(torch.zeros((1, 1), dtype=torch.long))).all())
            short = encode(['fire'], vocab, 8)
            mixed = encode(['fire', 'fire flood fire flood fire flood'], vocab, 8)
            torch.testing.assert_close(model(short)[0], model(mixed)[0], atol=1e-6, rtol=1e-5)

    def test_embedding_freeze_and_gradient(self):
        for frozen in [True, False]:
            config = dict(configurations()[0], frozen=frozen)
            model = PretrainedCNN(config, 4)
            model(torch.tensor([[2,3,0,0]])).sum().backward()
            if frozen:
                self.assertIsNone(model.embedding.weight.grad)
            else:
                self.assertGreater(float(model.embedding.weight.grad.abs().sum()), 0)
                self.assertEqual(float(model.embedding.weight.grad[0].abs().sum()), 0)

    def test_saved_validation_probabilities_and_preservation(self):
        out = ROOT/'experiments'/DEFAULT_RUN
        if not (out/'selection.json').exists():
            self.skipTest('Search not completed')
        loaded = load_selected(out)
        rows = pd.read_csv(ROOT/'data/processed/validation.csv')
        p = predict_selected(loaded, rows.text)
        np.testing.assert_allclose(p, np.load(out/'selected_validation_probabilities.npy'), atol=1e-6)
        for k, v in metrics(rows.target, p, loaded[0]['threshold']).items():
            self.assertAlmostEqual(v, loaded[0]['validation'][k])
        self.assertEqual(preserved_hashes(), json.loads((out/'input_hashes.json').read_text()))

    def test_saved_test_predictions_and_metrics(self):
        out = ROOT/'experiments'/DEFAULT_RUN
        if not (out/'summary.json').exists():
            self.skipTest('Final evaluation not completed')
        loaded = load_selected(out)
        rows = pd.read_csv(out/'test_predictions.csv')
        p = predict_selected(loaded, rows.text)
        np.testing.assert_allclose(p, rows.probability, atol=1e-6)
        np.testing.assert_array_equal(p >= loaded[0]['threshold'], rows.prediction)
        expected = json.loads((out/'summary.json').read_text())['comparison'][1]
        for k, v in metrics(rows.target, p, loaded[0]['threshold']).items():
            self.assertAlmostEqual(v, expected[k])


if __name__ == '__main__':
    unittest.main()
