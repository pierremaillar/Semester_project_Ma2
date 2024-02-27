import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd
import numpy as np

def compute_tsne(data, perplexity, n_iter=3000, n_iter_without_progress=100):
    """
    Computes t-Distributed Stochastic Neighbor Embedding (t-SNE) for the given data.

    Args:
        data (pd.DataFrame): Input data.
        perplexity (float): Perplexity parameter for t-SNE.
        n_iter (int): Maximum number of iterations.
        n_iter_without_progress (int): Number of iterations without progress to stop.

    Returns:
        np.ndarray: t-SNE projections.
    """
    tic = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=n_iter, n_iter_without_progress=n_iter_without_progress)
    tsne_proj = tsne.fit_transform(data)
    elapsed_time = time.time() - tic
    print(f"t-SNE done for perplexity {perplexity} in {elapsed_time} seconds")

    return tsne_proj



def plot_tsne(tsne, labels, category_col):
    """
    Plots t-SNE projections using Plotly Express.

    Args:
        tsne (np.ndarray): t-SNE projections.
        labels (pd.Series): Labels for coloring points in the plot.
        category_col (str): Name of the column to use for color-coding.

    Returns:
        None
    """
    #create dataframe with labels names
    tsne_df = pd.DataFrame({'Component 1': tsne[:, 0], 'Component 2': tsne[:, 1], 'labels': labels})
    mapping_dict = {index: category_col for index, category_col in enumerate(category_col)}
    tsne_df['labels'] = tsne_df['labels'].replace(mapping_dict)

    # Specify color map for each category
    color_map = {
        'Opisthokonta': 'rgba(255, 0, 0, 0.8)',   
        'Viridiplantae': 'rgba(0, 255, 0, 0.8)',     
        'Sar': 'rgba(0, 0, 255, 0.8)',              
        'Others': 'rgba(255, 255, 0, 0.8)'           
    }


    # Plot t-SNE with specified color map and legend order
    fig = px.scatter(
        tsne_df, x='Component 1', y='Component 2', color='labels',
        color_discrete_map=color_map,
        labels={'color': 'labels'}
    )

    fig.update_layout(
    height=500,  # set the height of the plot in pixels
    width=700    # set the width of the plot in pixels
    )
    
    fig.show()

    return 
