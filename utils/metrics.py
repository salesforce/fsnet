import numpy as np
import numexpr as ne
import pdb

def cumavg(m):
    cumsum= np.cumsum(m)
    return cumsum / np.arange(1, cumsum.size + 1)

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    
    return np.mean(np.abs(pred-true))
    #return ne.evaluate('sum(abs(pred-true))')/true.size
def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))
    #return ne.evaluate('sum(abs(pred-true)/true)')/true.shape[0]
    #return 0
    #return ne.evaluate('sum(abs(pred-true))')/(true.size)

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))
    #return 0

def metric(pred, true):
    
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    
    return mae,mse,rmse,mape,mspe
