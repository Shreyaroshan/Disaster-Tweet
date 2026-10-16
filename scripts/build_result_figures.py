"""Build measured four-model figures from the locked evaluation outputs."""
import sys
import json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from disaster_tweets.utils import ROOT,save_json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,roc_curve,auc


def main():
    folder=ROOT/'results/final_2026_10_16';figures=ROOT/'reports/final/figures';figures.mkdir(parents=True,exist_ok=True)
    records=json.loads((folder/'comparison.json').read_text());names=[r['model'].upper() for r in records]
    keys=['accuracy','precision','recall','f1'];x=np.arange(4)
    fig,ax=plt.subplots(figsize=(9,4.8))
    for i,(key,color) in enumerate(zip(keys,['#176b87','#4d8d83','#d49a38','#697b9d'])):
        bars=ax.bar(x+(i-1.5)*.19,[r[key]*100 for r in records],width=.18,label=key.upper() if key=='f1' else key.title(),color=color)
        ax.bar_label(bars,fmt='%.1f',padding=2,fontsize=7)
    ax.set_xticks(x,names);ax.set_ylim(0,100);ax.set_ylabel('Percent');ax.set_title('Four-model comparison on the same local test set');ax.legend(loc='upper center',ncol=4,frameon=False)
    ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(figures/'comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(8,7))
    for ax,row in zip(axes.flat,records):
        predictions=pd.read_csv(folder/row['model']/'test/predictions.csv');cm=confusion_matrix(predictions.target,predictions.prediction,labels=[0,1]);ax.imshow(cm,cmap='Blues')
        for (i,j),count in np.ndenumerate(cm):ax.text(j,i,str(count),ha='center',va='center',color='white' if count>cm.max()/2 else 'black',fontsize=14)
        ax.set(xticks=[0,1],yticks=[0,1],xlabel='Predicted',ylabel='Actual',title=row['model'].upper())
    fig.suptitle('0 non-disaster   1 disaster');fig.tight_layout();fig.savefig(figures/'confusion_matrices.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5))
    for row in records:
        predictions=pd.read_csv(folder/row['model']/'test/predictions.csv');fpr,tpr,_=roc_curve(predictions.target,predictions.probability)
        ax.plot(fpr,tpr,label=f"{row['model'].upper()} AUC {auc(fpr,tpr):.3f}")
    ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='False-positive rate',ylabel='True-positive rate',title='ROC curves');ax.legend(frameon=False);fig.tight_layout();fig.savefig(figures/'roc_curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(9,6))
    for ax,row in zip(axes.flat,records):
        selection=json.loads((folder/row['model']/'selection.json').read_text())
        history_root=ROOT/selection['search_history']
        paths=[]
        for metadata_path in history_root.glob('*/metadata.json'):
            metadata=json.loads(metadata_path.read_text())
            if metadata['seed']==42 and metadata['config']==selection['config']:
                paths.append(metadata_path.parent/'history.csv')
        if not paths and row['model']=='mlp':paths=list(history_root.glob('trial_15_seed42_history.csv'))
        if not paths and row['model']=='cnn':paths=list(history_root.glob('trial_03_seed42_history.csv'))
        if paths:
            h=pd.read_csv(paths[0]);column='validation_accuracy' if 'validation_accuracy' in h else 'accuracy'
            ax.plot(h.epoch,h[column],label='Validation accuracy');ax.set(xlabel='Epoch',ylabel='Accuracy',title=row['model'].upper());ax.legend(frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(figures/'validation_curves.png',dpi=180);plt.close(fig)
    # Deterministic sample for qualitative owner review; do not invent review labels.
    errors=[]
    for row in records:
        p=pd.read_csv(folder/row['model']/'test/errors.csv')
        p['confidence']=np.where(p.prediction==1,p.probability,1-p.probability)
        for error_type in [0,1]:
            sample=p[p.prediction==error_type].sort_values('confidence',ascending=False).head(3)
            for item in sample.to_dict('records'):errors.append({'model':row['model'],**item,'review_status':'requires team interpretation'})
    pd.DataFrame(errors).to_csv(folder/'error_review_sample.csv',index=False)
    print('Saved figures:',figures)


if __name__=='__main__':main()
