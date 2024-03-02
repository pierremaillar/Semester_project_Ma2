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
    # Sample a fraction of the dataset
    df_sampled = df.sample(frac=n)
    
    # Extract labels and data
    labels = df_sampled.iloc[:, 0].values.astype(int)
    data = df_sampled.iloc[:, 1:].values
    
    # Compute mutual information scores
    mutual_info = mutual_info_classif(data, labels)
    
    t = time.time() - tic
    print(f"Mutual info computed in {t} seconds")
    
    return mutual_info

