
import pandas as pd
import numpy as np

def generate_predictors_responses(df, k=5):
    mats = []
    for symbol, sub_df in df.groupby(level=0):
        n = sub_df.shape[0]
        for i in range(k, n):
            mats.append(sub_df.values[i-k:i+1])
    X = np.concatenate(mats, axis=1)
    
    return (X[:k].T, X[k])

def mse(y, yhat):
    return np.sqrt(np.mean(np.square(y - yhat)))
