"""
functions used to calculate the metrics for multi-label classification
Created on 27/08/2017
@author: Qian Wang
qian.wang@manchester.ac.uk
codes used in the paper: 
Qian Wang, Ke Chen, Multi-Label Zero-Shot Human Action Recognition via Joint Latent Embedding
"""
import numpy as np

def prf_cal(y_pred,y_true,k):
    """
    function to calculate top-k precision/recall/f1-score
    """
    GT=np.sum(y_true[y_true==1.])
    instance_num=y_true.shape[0]
    prediction_num=instance_num*k

    sort_indices = np.argsort(y_pred)
    sort_indices=sort_indices[:,::-1]
    static_indices = np.indices(sort_indices.shape)
    sorted_annotation= y_true[static_indices[0],sort_indices]
    top_k_annotation=sorted_annotation[:,0:k]
    TP=np.sum(top_k_annotation[top_k_annotation==1.])
    recall=TP/GT
    precision=TP/prediction_num
    f1=2.*recall*precision/(recall+precision)
    return precision, recall, f1

def cemap_cal(y_pred,y_true):
    """
    function to calculate L-MAP and I-MAP
    """
    nTest = y_true.shape[0]
    nLabel = y_true.shape[1]
    ap = np.zeros(nTest)
    for i in range(0,nTest):
        for j in range(0,nLabel):
            R = np.sum(y_true[i,:])
            if y_true[i,j]==1:
                r = np.sum(y_pred[i,:]>=y_pred[i,j])
                rb = np.sum(y_pred[i,np.nonzero(y_true[i,:])] >= y_pred[i,j])
                ap[i] = ap[i] + rb/(r*1.0)
        ap[i] = ap[i]/R
    imap = np.nanmean(ap)

    ap = np.zeros(nLabel)
    for i in range(0,nLabel):
        for j in range(0,nTest):
            R = np.sum(y_true[:,i])
            if y_true[j,i]==1:
                r = np.sum(y_pred[:,i] >= y_pred[j,i])
                rb = np.sum(y_pred[np.nonzero(y_true[:,i]),i] >= y_pred[j,i])
                ap[i] = ap[i] + rb/(r*1.0)
        ap[i] = ap[i]/R
    lmap = np.nanmean(ap)

    return lmap,imap
