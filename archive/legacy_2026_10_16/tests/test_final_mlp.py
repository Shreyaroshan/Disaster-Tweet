"""Final-project MLP: fold separation, local fitting, and artifact verification."""
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from final_mlp import (ROOT, DEFAULT_RUN, FinalMLP, configs, vocabulary_from, fit_features,
                       load, predict, metrics, preservation)
from project import encode


class FinalMLPTests(unittest.TestCase):
    def test_vocabulary_and_idf_are_fit_only_on_supplied_text(self):
        texts=['fire flood','fire rescue']
        vocab=vocabulary_from(texts)
        self.assertNotIn('earthquake',vocab)
        self.assertEqual(encode(['earthquake'],vocab,4)[0,0].item(),1)
        if not (ROOT/'data/embeddings/final_wiki100.npy').exists():
            self.skipTest('Prepare pretrained vectors first')
        fitted,weights,idf=fit_features(texts,['wiki100'])
        self.assertEqual(fitted,vocab)
        self.assertAlmostEqual(float(idf[vocab['fire']]),1.)
        self.assertAlmostEqual(float(idf[vocab['flood']]),float(np.log(3/2)+1),places=6)
        self.assertTrue(np.isfinite(weights).all())
        np.testing.assert_array_equal(weights[0],0)

    def test_padding_and_empty_input_are_safe(self):
        torch.set_num_threads(4)
        for use_idf in [False,True]:
            config=dict(configs()[0],idf_pool=use_idf)
            model=FinalMLP(config,4,idf=np.array([0.,1.,2.,3.],dtype=np.float32)).eval()
            x=torch.tensor([[2,3,0],[1,0,0],[0,0,0]])
            a=model(x)
            self.assertTrue(torch.isfinite(a).all())
            torch.testing.assert_close(a,model(torch.nn.functional.pad(x,(0,7))))

    def test_cv_folds_are_disjoint_and_exclude_original_validation_test(self):
        out=ROOT/'experiments'/DEFAULT_RUN
        if not (out/'folds.json').exists(): self.skipTest('Search not started')
        train=set(pd.read_csv(ROOT/'data/processed/train.csv').id)
        excluded=set(pd.read_csv(ROOT/'data/processed/validation.csv').id)|set(pd.read_csv(ROOT/'data/processed/test.csv').id)
        folds=json.loads((out/'folds.json').read_text())
        assessed=[]
        for fold in folds:
            a,b,c=[set(fold[k]) for k in ['fitting_ids','stopping_ids','assessment_ids']]
            self.assertFalse(a&b or a&c or b&c)
            self.assertEqual(a|b|c,train)
            self.assertFalse((a|b|c)&excluded)
            assessed+=list(c)
        self.assertEqual(len(assessed),len(set(assessed)))
        self.assertEqual(set(assessed),train)
        # Inspect actual saved fold vocabularies, not only index separation.
        frame=pd.read_csv(ROOT/'data/processed/train.csv').set_index('id')
        records=json.loads((out/'runs.json').read_text()) if (out/'runs.json').exists() else []
        for record in records:
            cp_path=out/(record['name']+'.pt')
            fold=int(cp_path.stem.rsplit('fold',1)[1])
            cp=torch.load(cp_path,map_location='cpu',weights_only=True)
            expected=vocabulary_from(frame.loc[folds[fold]['fitting_ids']].text.tolist())
            self.assertEqual(cp['vocabulary'],expected)

    def test_saved_model_reproduces_validation_and_test(self):
        out=ROOT/'experiments'/DEFAULT_RUN
        if not (out/'summary.json').exists(): self.skipTest('Final evaluation not completed')
        torch.set_num_threads(4)
        loaded=load(out)
        for part,filename in [('validation','validation_predictions.csv'),('test','test_predictions.csv')]:
            actual=pd.read_csv(ROOT/f'data/processed/{part}.csv')
            saved=pd.read_csv(out/filename)
            self.assertTrue(actual.id.equals(saved.id))
            p=predict(loaded,actual.text.tolist())
            np.testing.assert_allclose(p,saved.probability,atol=1e-6)
            expected=loaded[0]['validation'] if part=='validation' else json.loads((out/'summary.json').read_text())['comparison'][1]
            for k,v in metrics(actual.target,p,loaded[0]['threshold']).items(): self.assertAlmostEqual(v,expected[k])
        self.assertEqual(preservation(),json.loads((out/'input_hashes.json').read_text()))


if __name__=='__main__': unittest.main()
