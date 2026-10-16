import argparse
from disaster_tweets.evaluation import evaluate_all
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='final_2026_10_16');args=parser.parse_args();evaluate_all(args.run)
