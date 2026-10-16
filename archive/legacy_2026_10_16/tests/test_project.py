import json, sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from project import *

class PipelineTests(unittest.TestCase):
    def test_clean_preserves_negation_and_hashtag_word(self):
        self.assertEqual(clean("NOT a #disaster @bob https://a.co &amp; safe"),'not a disaster usertoken urltoken safe')
    def test_no_split_leakage_and_train_only_vocabulary(self):
        parts={k:pd.read_csv(ROOT/f'data/processed/{k}.csv') for k in ['train','validation','test']}
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            self.assertFalse(set(parts[a].clean_text)&set(parts[b].clean_text))
            self.assertFalse(set(parts[a].id)&set(parts[b].id))
        vocab=json.loads((ROOT/'data/processed/vocabulary.json').read_text())
        self.assertTrue((set(vocab)-{'<PAD>','<UNK>'}).issubset(set(' '.join(parts['train'].clean_text).split())))
    def test_unknown_empty_and_padding(self):
        c=CONFIG; vocab={'<PAD>':0,'<UNK>':1,'fire':2,'no':3}
        x=encode(['','fire','no fire','unseen'],vocab,64)
        self.assertEqual(x[0,0].item(),1)
        for cls in MODELS.values():
            model=cls(len(vocab),c).eval()
            a=model(x); b=model(torch.nn.functional.pad(x,(0,10)))
            self.assertTrue(torch.isfinite(a).all())
            torch.testing.assert_close(a,b)
    def test_checkpoint_reproduces_saved_predictions(self):
        for name in MODELS:
            rows=pd.read_csv(ROOT/f'reports/{name}_test_predictions.csv').head(3)
            for row in rows.itertuples():
                self.assertAlmostEqual(predict(name,row.text)['disaster_probability'],row.probability,places=5)
    def test_report_metrics_match_predictions(self):
        results=json.loads((ROOT/'reports/results.json').read_text())
        for result in results:
            rows=pd.read_csv(ROOT/f'reports/{result["model"]}_test_predictions.csv')
            for key,value in metrics(rows.target,rows.probability).items(): self.assertAlmostEqual(value,result[key])

if __name__=='__main__': unittest.main()
