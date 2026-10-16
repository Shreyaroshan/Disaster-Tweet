"""Shared data, padding and artifact checks for the four-model implementation."""
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from disaster_tweets.preprocessing import clean_text,fit_vocabulary,encode,load_data
from disaster_tweets.models import build_model
from disaster_tweets.utils import ROOT,sha256
from disaster_tweets.training import configure_runtime,selected_probabilities
from disaster_tweets.metrics import classification_metrics


class SharedPipelineTests(unittest.TestCase):
    def setUp(self):configure_runtime()

    def require_models(self,names):
        from importlib.util import find_spec
        missing=[name for name in names if find_spec(f"disaster_tweets.models.{name}") is None]
        if missing:self.skipTest("Awaiting member model uploads: "+", ".join(missing))

    def test_cleaning_preserves_negation_and_hashtag_words(self):
        self.assertEqual(clean_text('NOT a #disaster @bob https://a.co &amp; safe'),'not a disaster usertoken urltoken safe')
        self.assertEqual(encode([''],{'<PAD>':0,'<UNK>':1},4).tolist(),[[1,0,0,0]])

    def test_no_data_leakage_and_training_only_vocabulary(self):
        parts,vocab,manifest=load_data(include_test=True)
        self.assertEqual(vocab,fit_vocabulary(parts['train'].text.tolist()))
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            self.assertFalse(set(parts[a].id)&set(parts[b].id))
            self.assertFalse(set(parts[a].text.map(clean_text))&set(parts[b].text.map(clean_text)))
        self.assertEqual([len(parts[k]) for k in ['train','validation','test']],[4770,1022,1023])

    def test_all_architectures_ignore_right_padding_and_handle_short_text(self):
        self.require_models(['mlp','cnn','lstm','transformer'])
        for name in ['mlp','cnn','lstm','transformer']:
            config=json.loads((ROOT/f'configs/{name}.json').read_text())
            if isinstance(config,list):config=config[0]
            model=build_model(name,5,config).eval()
            x=torch.tensor([[2,3,0,0,0],[1,0,0,0,0],[2,3,4,2,3]])
            a=model(x);b=model(torch.nn.functional.pad(x,(0,10)))
            torch.testing.assert_close(a,b,atol=2e-6,rtol=1e-5)
            self.assertTrue(torch.isfinite(a).all())
            torch.testing.assert_close(model(x[:1])[0],a[0],atol=2e-6,rtol=1e-5)
            self.assertTrue(torch.isfinite(model(torch.zeros((1,4),dtype=torch.long))).all())

    def test_transformer_uses_distinct_layer_initializations(self):
        self.require_models(['transformer'])
        configs=json.loads((ROOT/'configs/search/transformer.json').read_text());config=next(c for c in configs if c['layers']==2)
        model=build_model('transformer',5,config)
        a=model.encoder.layers[0].self_attn.in_proj_weight;b=model.encoder.layers[1].self_attn.in_proj_weight
        self.assertFalse(torch.equal(a,b))

    def test_frozen_embeddings_have_no_gradient(self):
        self.require_models(['mlp','cnn','lstm','transformer'])
        for name in ['mlp','cnn','lstm','transformer']:
            config=json.loads((ROOT/f'configs/{name}.json').read_text());config=config[0] if isinstance(config,list) else config
            config=dict(config,frozen=True)
            model=build_model(name,5,config)
            model(torch.tensor([[2,3,0,0,0]])).sum().backward()
            self.assertIsNone(model.embedding.weight.grad)

    def test_final_saved_predictions_and_reported_metrics(self):
        self.require_models(['mlp','cnn','lstm','transformer'])
        root=ROOT/'results/final_2026_10_16'
        if not (root/'comparison.json').exists():self.skipTest('Final evaluation not run yet')
        data,_,_=load_data(include_test=True)
        for record in json.loads((root/'comparison.json').read_text()):
            name=record['model'];selection=json.loads((root/name/'selection.json').read_text())
            saved=pd.read_csv(root/name/'test/predictions.csv')
            actual=selected_probabilities(selection,data['test'].text.tolist())
            self.assertTrue(saved.id.equals(data['test'].id))
            np.testing.assert_allclose(actual,saved.probability,atol=1e-6)
            np.testing.assert_array_equal(actual>=selection['threshold'],saved.prediction)
            for k,v in classification_metrics(saved.target,actual,selection['threshold']).items():self.assertAlmostEqual(v,record[k])


if __name__=='__main__':unittest.main()
