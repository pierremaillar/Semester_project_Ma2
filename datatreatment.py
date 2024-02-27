import pandas as pd
import numpy as np

def extract_word_at_position(X, p):
    """Extract a word at a specific position in a comma-separated string.

    Args:
        X (str): Comma-separated string.
        p (int): Position to extract.

    Returns:
        str: Word at the specified position.
    """
    split_values = X.split(",")
    if p < len(split_values):
        return split_values[p]
    else:
        return ""  # Return empty string if index is out of range


def length_taxo(X):
    """Calculate the length of a taxonomic string.

    Args:
        X (str): Taxonomic string.

    Returns:
        int: Length of the taxonomic string.
    """
    return len(X.split(","))


def importing_data(adresse):
    """Import data from a CSV file and perform column renaming.

    Args:
        adresse (str): File path of the CSV file.

    Returns:
        pd.DataFrame: Imported data with column renaming.
    """
    data = pd.read_csv(adresse)
    data.rename(columns={'Entry Name': 'Entry'}, inplace=True)
    data.rename(columns={'Taxonomic lineage': 'Taxonomic'}, inplace=True)

    return data


def get_data(data, level, number_category, type_classification="notspecified"):
    """Preprocess and filter data based on taxonomic information.

    Args:
        data (pd.DataFrame): Input data with taxonomic information.
        level (int): Taxonomic level to consider.
        number_category (int): Number of categories to keep.
        type_classification (str): Type of classification (default is "notspecified").

    Returns:
        pd.DataFrame: Processed data with specified number of categories.
        list: Name of the categories.
    """
    data['nb_niv'] = data['Taxonomic'].apply(lambda x: length_taxo(x))
    
    # Use .loc to avoid SettingWithCopyWarning
    data = data.loc[data['nb_niv'] >= 3].copy()

    # Extract the specified taxonomic level
    data.loc[:, f'level {level}'] = data['Taxonomic'].apply(lambda x: extract_word_at_position(x, level - 1))
    
    if type_classification != "notspecified":
        data.loc[:, 'previous level'] = data['Taxonomic'].apply(lambda x: extract_word_at_position(x, level - 2))
        data.loc[:, 'previous level'] = data['previous level'].str.replace(' ', '')
        specify_data = data[data['previous level'] == type_classification].copy()
    else:
        specify_data = data.copy()
    
    specify_data.loc[:, f'level {level}'] = specify_data[f'level {level}'].str.replace(' ', '')

    name_category = specify_data[f'level {level}'].value_counts().head(number_category - 1).index.tolist()
    specify_data.loc[:, f'level {level}'] = specify_data[f'level {level}'].apply(
        lambda x: x if x in name_category else 'Others')
    name_category.append('Others')

    specify_data = specify_data.drop(['nb_niv', 'Taxonomic', 'Organism', 'Entry'], axis=1)
    if type_classification != "notspecified":
        specify_data = specify_data.drop(['previous level'], axis=1)

    specify_data = specify_data.reset_index(drop=True)
    specify_data.loc[:, f'level {level}'] = specify_data[f'level {level}'].astype("category")

    for abc in name_category:
        print(f'number of {abc} : ')
        print({specify_data[f'level {level}'].value_counts()[abc]})
        print("---------")

    return specify_data, name_category


def encode01(data_to_encode):
    """Encode a column of sequences into binary columns (0/1 encoding).

    Args:
        data_to_encode (pd.DataFrame): Dataframe with a column containing the sequences. (here 'Hsp70_sequence')

    Returns:
        pd.DataFrame: Encoded data with binary columns.
    """
    #split the sequence into different columns
    split_sequence = data_to_encode['Hsp70_sequence'].apply(lambda x: pd.Series(list(x)))

    # Rename the columns to pos_1, pos_2, ...
    split_sequence.columns = [f'pos_{i+1}' for i in range(split_sequence.shape[1])]

    #encodes the sequences
    dummy_cols = pd.get_dummies(split_sequence.filter(regex='^pos_'))
    encoded_data = pd.concat([data_to_encode, dummy_cols], axis=1)
    
    #drop the full sequence column
    encoded_data = encoded_data.drop(['Hsp70_sequence'], axis=1)

    print(f'shape of dataframe : {encoded_data.shape}')

    return encoded_data


def category_to_int(data, name_category):
    """Convert categorical labels to integer labels based on a mapping.

    Args:
        data (pd.DataFrame): Dataframe with categorical labels.
        name_category (list): List of category names.

    Returns:
        pd.DataFrame: Dataframe with integer-encoded labels.
    """
    map_dict = {label: index for index, label in enumerate(name_category)}
    data.iloc[:, 0] = data.iloc[:, 0].replace(map_dict)

    return data