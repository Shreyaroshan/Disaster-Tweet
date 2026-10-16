import argparse
import json
from disaster_tweets.utils import ROOT
from disaster_tweets.training import search

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True,choices=['mlp','cnn','lstm','transformer']);parser.add_argument('--run',default='final_2026_10_16');parser.add_argument('--config');args=parser.parse_args()
    path=ROOT/(args.config or f'configs/{args.model}.json');value=json.loads(path.read_text());configs=value if isinstance(value,list) else [value]
    search(args.model,args.run,configs)
