import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix


def classification_metrics(labels,probabilities,threshold=.5):
    labels=np.asarray(labels);probabilities=np.asarray(probabilities);pred=probabilities>=threshold
    return {'accuracy':float(accuracy_score(labels,pred)),'precision':float(precision_score(labels,pred,zero_division=0)),
            'recall':float(recall_score(labels,pred,zero_division=0)),'f1':float(f1_score(labels,pred,zero_division=0)),
            'roc_auc':float(roc_auc_score(labels,probabilities)) if len(np.unique(labels))==2 else None}


def confusion_counts(labels,probabilities,threshold=.5):
    tn,fp,fn,tp=confusion_matrix(labels,np.asarray(probabilities)>=threshold,labels=[0,1]).ravel()
    return {name:int(value) for name,value in zip(['true_negative','false_positive','false_negative','true_positive'],[tn,fp,fn,tp])}
