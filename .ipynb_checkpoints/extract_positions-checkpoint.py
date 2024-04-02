from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd



def extract_positions(dataset, sequence, fixed_value, window = 0):
    """
    Extract positions in the sequence and corresponding columns from the dataset.

    Args:
        dataset (pd.DataFrame): Input dataset.
        sequence (list): List representing the sequence.
        fixed_value (float): Threshold value for positions to keep.

    Returns:
        pd.DataFrame: Filtered dataset containing selected columns.
        list: List of positions in the sequence.
    """
    # Extract positions in the sequence above the fixed value
    positions = [index for index, value in enumerate(sequence) if value > fixed_value]
    
    #add all the positions near (-and+ window)the positions selected
    if window != 0:
        positions_temp=[]
        for num in positions:
            positions_temp.extend([i for i in range(num - window , num + window+1)])

        # Remove duplicates and sort the result list
        positions = sorted(list(set(positions_temp)))

    # Convert positions to column names format
    new_positions = ['_' + str(element) + '_' for element in positions]

    # Get column names from the dataset
    column_names = dataset.columns.tolist()

    # Extract column positions in the dataset corresponding to selected positions
    values_to_extract = new_positions
    positions_tot = [pos for pos, name in enumerate(column_names) if any(value in name for value in values_to_extract)]

    # Extract the selected columns along with the first column (assuming it contains labels)
    dataset_filtered = dataset.iloc[:, [0] + positions_tot]

    return positions, dataset_filtered