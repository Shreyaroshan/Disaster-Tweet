import argparse
import json
from disaster_tweets.utils import ROOT
from disaster_tweets.training import configure_runtime,selected_probabilities
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True,choices=['mlp','cnn','lstm','transformer']);parser.add_argument('--run',default='final_2026_10_16');parser.add_argument('--text',required=True);args=parser.parse_args()
    configure_runtime();selection=json.loads((ROOT/f'results/{args.run}/{args.model}/selection.json').read_text());p=float(selected_probabilities(selection,[args.text])[0]);t=selection['threshold']
    print(json.dumps({'model':args.model,'disaster_probability':p,'threshold':t,'prediction':int(p>=t),'label':'disaster' if p>=t else 'non-disaster'},indent=2))
