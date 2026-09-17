"""Behavioral checks for the improved MLP representations and saved deployment."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from improve_mlp import (ROOT, DEFAULT_RUN, GloveMLP, make_vectorizer,
                         load_selected, selected_probabilities, metrics, input_hashes)
from project import encode
import numpy as np
import pandas as pd
import torch


class ImprovedMLPTests(unittest.TestCase):
    def test_pooling_ignores_padding_and_handles_empty_text(self):
        vocab = {'<PAD>': 0, '<UNK>': 1, 'fire': 2}
        x = encode(['fire', '', 'unknown'], vocab, 6)
        for pool in ['mean', 'mean_max']:
            c = dict(pool=pool, frozen=False, dropout=.3, hidden=8)
            model = GloveMLP(c, len(vocab)).eval()
            a = model(x)
            b = model(torch.nn.functional.pad(x, (0, 9)))
            torch.testing.assert_close(a, b)
            self.assertTrue(torch.isfinite(a).all())
            self.assertTrue(torch.isfinite(model(torch.zeros_like(x))).all())

    def test_frozen_embeddings_stay_fixed_and_finetuning_has_gradient(self):
        for frozen in [False, True]:
            c = dict(pool='mean', frozen=frozen, dropout=0., hidden=8)
            model = GloveMLP(c, 4)
            before = model.embedding.weight.detach().clone()
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=.01)
            loss = model(torch.tensor([[2, 3, 0]])).sum()
            loss.backward()
            optimizer.step()
            if frozen:
                torch.testing.assert_close(before, model.embedding.weight)
                self.assertIsNone(model.embedding.weight.grad)
            else:
                self.assertGreater(float(model.embedding.weight.grad.abs().sum()), 0.)
                self.assertFalse(torch.equal(before, model.embedding.weight))

    def test_tfidf_transform_does_not_learn_validation_words(self):
        vectorizer = make_vectorizer(False)
        vectorizer.fit(['fire road', 'fire rain', 'rain road'])
        original = dict(vectorizer.vocabulary_)
        vectorizer.transform(['validationonlyword earthquake'])
        self.assertEqual(vectorizer.vocabulary_, original)
        self.assertNotIn('validationonlyword', original)

    def test_saved_selection_reproduces_validation_and_preserves_inputs(self):
        out = ROOT / 'experiments' / DEFAULT_RUN
        if not (out / 'selection.json').exists():
            self.skipTest('Run search first for saved-model verification')
        torch.set_num_threads(4)
        loaded = load_selected(out)
        validation = pd.read_csv(ROOT / 'data/processed/validation.csv')
        actual = selected_probabilities(loaded, validation.text.tolist())
        np.testing.assert_allclose(actual, np.load(out / 'selected_validation_probabilities.npy'), atol=1e-6)
        result = metrics(validation.target, actual, loaded[0]['threshold'])
        for key, value in result.items():
            self.assertAlmostEqual(value, loaded[0]['validation_selected_threshold'][key])
        self.assertEqual(input_hashes(), json.loads((out / 'input_hashes.json').read_text()))

    def test_saved_test_metrics_and_inference_batch_invariance(self):
        out = ROOT / 'experiments' / DEFAULT_RUN
        if not (out / 'summary.json').exists():
            self.skipTest('Run final evaluation first')
        torch.set_num_threads(4)
        loaded = load_selected(out)
        predictions = pd.read_csv(out / 'test_predictions.csv')
        actual = selected_probabilities(loaded, predictions.text.tolist())
        np.testing.assert_allclose(actual, predictions.probability, atol=1e-6)
        for row in predictions.head(3).itertuples():
            single = selected_probabilities(loaded, [row.text])[0]
            self.assertAlmostEqual(float(single), row.probability, places=5)
        expected = json.loads((out / 'summary.json').read_text())['comparison'][2]
        for key, value in metrics(predictions.target, actual, loaded[0]['threshold']).items():
            self.assertAlmostEqual(value, expected[key])


if __name__ == '__main__':
    unittest.main()
