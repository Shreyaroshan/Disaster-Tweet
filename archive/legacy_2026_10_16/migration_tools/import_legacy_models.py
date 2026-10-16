"""Migrate selected historical checkpoints without changing tensors or predictions."""
import sys
import json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from disaster_tweets.utils import ROOT,save_json,sha256
from disaster_tweets.preprocessing import load_data,encode
from disaster_tweets.training import configure_runtime,load_checkpoint,probabilities


def main():
    configure_runtime();parts,vocab,_=load_data()
    plans=[('mlp','mlp_improved_2026_09_18','trial_15_seed42',.49),('cnn','cnn_improved_2026_09_18','trial_03_seed42',.5)]
    for model,run,trial,threshold in plans:
        source=ROOT/'experiments'/run
        cp=torch.load(source/f'{trial}.pt',map_location='cpu',weights_only=True)
        original=cp['config'];canonical=json.loads((ROOT/f'configs/{model}.json').read_text())
        assert cp['vocabulary']==vocab
        destination=ROOT/f'artifacts/models/{model}/historical.pt';destination.parent.mkdir(parents=True,exist_ok=True)
        converted={'format_version':1,'model':model,'config':canonical,'seed':42,'vocabulary':vocab,'state_dict':cp['state_dict'],'threshold':threshold,'max_length':64,
                   'data_manifest_sha256':sha256(ROOT/'data/processed/manifest.json'),'source_checkpoint_sha256':sha256(source/f'{trial}.pt')}
        torch.save(converted,destination)
        loaded,network=load_checkpoint(destination)
        p=probabilities(network,encode(parts['validation'].text.tolist(),vocab,64))
        if model=='mlp':
            prior=pd.read_csv(source/f'{trial}_history.csv')
            from disaster_tweets.metrics import classification_metrics
            validation=classification_metrics(parts['validation'].target,p,threshold)
            expected=json.loads((source/'selection.json').read_text())['validation_selected_threshold']
        else:
            from disaster_tweets.metrics import classification_metrics
            validation=classification_metrics(parts['validation'].target,p,threshold)
            expected=json.loads((source/'selection.json').read_text())['validation']
        for k in ['accuracy','precision','recall','f1']:assert abs(validation[k]-expected[k])<1e-10
        metadata=json.loads((source/f'{trial}.json').read_text())
        selection={'model':model,'config':canonical,'threshold':threshold,'ensemble':False,'validation':validation,
                   'members':[{'path':str(destination.relative_to(ROOT)),'sha256':sha256(destination),'seed':42}],
                   'parameters_per_model':sum(p.numel() for p in network.parameters()),'training_seconds':metadata['seconds'],
                   'selected_fit_seconds':metadata['seconds'],'search_fits':None,'origin':'Historical September validation-selected checkpoint; no tensor changes.',
                   'historical_selection_metric':'MLP architecture screened by F1 with validation accuracy refinement; CNN accuracy primary.',
                   'search_history':f'archive/legacy_2026_10_16/experiments/{run}'}
        save_json(ROOT/f'results/retained/{model}/selection.json',selection)
        print(model,'migration predictions verified',validation)


if __name__=='__main__':main()
