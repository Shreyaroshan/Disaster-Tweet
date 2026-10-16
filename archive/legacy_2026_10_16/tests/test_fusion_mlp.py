"""Behavioral tests for lexical/semantic MLP and its saved artifacts."""
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import numpy as np
import pandas as pd
import torch
from fusion_mlp import ROOT,DEFAULT_RUN,FusionMLP,configs,vectorizer,load,predict,metrics
from final_mlp import preservation,vocabulary_from


class FusionMLPTests(unittest.TestCase):
    def test_padding_and_empty_inputs(self):
        torch.set_num_threads(4)
        model=FusionMLP(configs()[0],4,5).eval()
        x=torch.tensor([[2,3,0],[1,0,0],[0,0,0]])
        lexical=torch.zeros(3,5)
        actual=model(x,lexical)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual,model(torch.nn.functional.pad(x,(0,10)),lexical))

    def test_lexical_vocabulary_does_not_learn_transform_data(self):
        vec=vectorizer(False)
        vec.fit(['fire flood','fire rescue','flood rescue'])
        before=dict(vec.vocabulary_)
        vec.transform(['unseenvalidationtoken earthquake'])
        self.assertEqual(before,vec.vocabulary_)
        self.assertNotIn('unseenvalidationtoken',vec.vocabulary_)

    def test_saved_fold_features_fit_on_inner_training_only(self):
        out=ROOT/'experiments'/DEFAULT_RUN
        if not (out/'runs.json').exists(): self.skipTest('Search has not started')
        records=json.loads((out/'runs.json').read_text())
        frame=pd.read_csv(ROOT/'data/processed/train.csv').set_index('id')
        folds=json.loads((out/'folds.json').read_text())
        for record in records:
            if record['seed']!=42 or record['fold']!=0: continue
            fitting=frame.loc[folds[0]['fitting_ids']]
            cp=torch.load(out/(record['name']+'.pt'),map_location='cpu',weights_only=True)
            self.assertEqual(cp['vocabulary'],vocabulary_from(fitting.text.tolist()))
            actual=joblib.load(out/(record['name']+'_vectorizer.joblib'))
            expected=vectorizer(record['config']['characters']).fit(fitting.text)
            np.testing.assert_array_equal(actual.get_feature_names_out(),expected.get_feature_names_out())
            a=actual.transform(['fire flood rescue']).toarray();b=expected.transform(['fire flood rescue']).toarray()
            np.testing.assert_allclose(a,b,atol=1e-7)

    def test_checkpoint_predictions_and_preservation(self):
        out=ROOT/'experiments'/DEFAULT_RUN
        if not (out/'summary.json').exists(): self.skipTest('Evaluation not complete')
        loaded=load(out)
        for split in ['validation','test']:
            frame=pd.read_csv(ROOT/f'data/processed/{split}.csv')
            saved=pd.read_csv(out/f'{split}_predictions.csv')
            self.assertTrue(frame.id.equals(saved.id))
            p=predict(loaded,frame.text)
            np.testing.assert_allclose(p,saved.probability,atol=1e-6)
            expected=loaded[0]['validation'] if split=='validation' else json.loads((out/'summary.json').read_text())['comparison'][1]
            for k,v in metrics(frame.target,p,loaded[0]['threshold']).items(): self.assertAlmostEqual(v,expected[k])
        self.assertEqual(preservation(),json.loads((out/'input_hashes.json').read_text()))


if __name__=='__main__': unittest.main()
