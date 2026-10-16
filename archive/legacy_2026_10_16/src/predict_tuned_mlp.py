"""Predict using the tuned MLP and its saved validation-selected threshold."""
import argparse
import json
import torch
from project import ROOT, encode, probabilities
from tune_mlp import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text', required=True)
    parser.add_argument('--checkpoint', default=str(ROOT / 'experiments/mlp_tuning_2026_09_17/mlp_tuned.pt'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint, model = load_model(args.checkpoint)
    tokens = encode([args.text], checkpoint['vocabulary'], checkpoint['config']['max_length'])
    probability = float(probabilities(model, tokens)[0])
    threshold = checkpoint['threshold']
    print(json.dumps({'model': 'tuned_mlp', 'disaster_probability': probability,
                      'threshold': threshold, 'prediction': int(probability >= threshold),
                      'label': 'disaster' if probability >= threshold else 'non-disaster'}, indent=2))


if __name__ == '__main__':
    main()
