"""Conditional paired uncertainty estimates; not independent generalization tests."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from disaster_tweets.utils import ROOT,save_json
from disaster_tweets.metrics import classification_metrics
import pandas as pd
import numpy as np


def main():
    root=ROOT/'results/final_2026_10_16'
    a=pd.read_csv(root/'mlp/test/predictions.csv');b=pd.read_csv(root/'lstm/test/predictions.csv')
    assert a.id.equals(b.id);rng=np.random.default_rng(20261016);deltas=[]
    for _ in range(2000):
        idx=rng.integers(0,len(a),len(a));aa=classification_metrics(a.target.iloc[idx],a.probability.iloc[idx],.49);bb=classification_metrics(b.target.iloc[idx],b.probability.iloc[idx],.48)
        deltas.append([bb[k]-aa[k] for k in ['accuracy','f1']])
    intervals={k:np.percentile(np.array(deltas)[:,i],[2.5,97.5]).tolist() for i,k in enumerate(['accuracy','f1'])}
    save_json(root/'paired_mlp_lstm.json',{'direction':'LSTM minus MLP','bootstrap_samples':2000,'95_percent_intervals':intervals,
              'caveat':'Conditional on fitted models; ignores seed/search uncertainty and tweet-event correlations. Reused local test; exploratory.'})
    print(intervals)


if __name__=='__main__':main()
