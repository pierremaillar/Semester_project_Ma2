import sklearn
import time
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

def compute_mutual_info(df, n):
    """Compute mutual information scores and return the result.

    Args:
        df (pd.DataFrame): Input data.
        n (float): Fraction of the dataset to use for computing mutual information.

    Returns:
        numpy.ndarray: Mutual information scores.
    """
    tic = time.time()
    #choose a subset of the data
    ids = np.arange(df.shape[0])
    np.random.shuffle(ids)
    p = round(df.shape[0] * n)
    x_shuffle = df.iloc[ids, :]
    df_batch = x_shuffle.iloc[:p, :]
    df_batch.reset_index(drop=True, inplace=True)
    #extracting the labels
    labels = df_batch.iloc[:, 0].values
    labels = labels.astype(int)
    #droping the columns of the dataset that contains the labels
    data = df_batch.drop(df.columns[0], axis=1).values
    #compute the mutual information scores  (using only a fraction n of the all dataset)
    mutual_info = sklearn.feature_selection.mutual_info_classif(data, labels)
    t = time.time() - tic
    print(f"Mutual info computed in {t} seconds")
    return mutual_info

