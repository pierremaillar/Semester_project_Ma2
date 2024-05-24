from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd



def extract_positions(dataset,pos_list):
    """
    Extract positions in the sequence and corresponding columns from the dataset.

    Args:
        dataset (pd.DataFrame): Input dataset.
        pos_list (list of integer): List of the positions to keep.

    Returns:
        pd.DataFrame: Filtered dataset containing selected columns.
    """

    selected_columns = dataset[['Labels']]
    
    # Iterate over each position in the pos_list
    for pos in pos_list:
        # Construct the pattern to match column names
        pattern = f'^pos_{pos}_'
        # Select columns that match the pattern for the current position
        columns_for_pos = dataset.filter(regex=pattern)
        # Concatenate the selected columns to the DataFrame
        selected_columns = pd.concat([selected_columns, columns_for_pos], axis=1)
        
    return selected_columns