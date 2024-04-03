import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

def extract_taxonomic_info(dataframe, threshold):
    # Initialize an empty dictionary to store counts
    taxonomic_counts = {}
    
    # Iterate through each row in the dataframe
    for index, row in dataframe.iterrows():
        # Extract the taxonomic information from the "Taxonomic" column
        taxonomic_info = row['Taxonomic']
        
        # Split the taxonomic information into individual names
        taxonomic_names = taxonomic_info.split(',')
        
        # Iterate through each name and update the counts
        for name in taxonomic_names:
            # Remove leading/trailing whitespaces
            name = name.strip()
            
            # Update the counts dictionary
            if name in taxonomic_counts:
                taxonomic_counts[name] += 1
            else:
                taxonomic_counts[name] = 1
                
    # Remove entries from the dictionary with counts below the threshold
    taxonomic_counts_cleaned = {key: value for key, value in taxonomic_counts.items() if value >= threshold}
    
    # Convert the dictionary into a pandas DataFrame for easier manipulation
    taxonomic_counts_df = pd.DataFrame(list(taxonomic_counts_cleaned.items()), columns=['Taxonomic Name', 'Count'])
    
    return taxonomic_counts_df




def create_sankey_diagram(dataframe, num_levels, threshold, node_pad=12, node_thickness=10, node_label_size=12):
    # Initialize lists to store source, target, and count for the Sankey diagram
    sources = []
    targets = []
    counts = []
    nbr_sources = []
    pos_ys=[]

    for index, row in dataframe.iterrows():
        # Extract the taxonomic information from the "Taxonomic" column
        taxonomic_info = row['Taxonomic']

        # Split the taxonomic information into individual names
        taxonomic_names = taxonomic_info.split(',')[:num_levels]

        # Iterate through each name and update the counts
        for i in range(len(taxonomic_names) - 1):
            # Remove leading/trailing whitespaces
            source = taxonomic_names[i].strip()
            nbr_source = i
            pos_y= i
            target = taxonomic_names[i + 1].strip()

            # Append source, target, and count to the lists
            sources.append(source)
            targets.append(target)
            nbr_sources.append(nbr_source)
            pos_ys.append(pos_y)
            counts.append(1) 


    scaler = MinMaxScaler()
    normalized = scaler.fit_transform([[x] for x in nbr_sources])
    nbr_sources = normalized.flatten()

    scaler = MinMaxScaler()
    normalized = scaler.fit_transform([[x] for x in pos_ys])
    pos_ys = normalized.flatten()

    sankey_df = pd.DataFrame({'Source': sources, 'Target': targets, 'Count': counts, 'Nbr Sources': nbr_sources, 'pos_y': pos_ys})

    # Aggregate counts based on source and target
    sankey_df = sankey_df.groupby(['Source', 'Target', 'Nbr Sources','pos_y']).size().reset_index(name='Count')

    # Filter out entries with counts below the threshold
    sankey_df = sankey_df[sankey_df['Count'] >= threshold]

    # Create string to integer dictionary
    string_to_integer_dict = {string: index for index, string in enumerate(set(sources) | set(targets))}

    # Map strings to integers
    sankey_df["Source"] = sankey_df["Source"].map(string_to_integer_dict)
    sankey_df["Target"] = sankey_df["Target"].map(string_to_integer_dict)
    
    # Create a list to store RGB colors
    colors = []
    for i in range(len(sankey_df)):
        red = np.random.randint(0, 256)
        green = np.random.randint(0, 256)
        blue = np.random.randint(0, 256)
        opacity = np.random.randint(1, 100)
        colors.append(f'rgba({red}, {green}, {blue},{opacity/100})')
    
    sankey_df['Colors'] = colors
    print(sankey_df)
    
    # Create a Plotly Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        arrangement='snap',
        
        node=dict(
            pad=node_pad,
            thickness=node_thickness,
            line=dict(color="black", width=0.5),
            label=list(string_to_integer_dict.keys()),
            color=sankey_df['Colors'],
            x=sankey_df['Nbr Sources'],
            y=sankey_df['pos_y']
        ),
        link=dict(
            source=sankey_df["Source"],  # Source nodes
            target=sankey_df["Target"],  # Target nodes
            value=sankey_df['Count'],     # Values
            color=sankey_df['Colors']
        ))])

    # Update layout
    fig.update_layout(
        title_text="Taxonomic Sankey Diagram",
        font=dict(size=node_label_size)  # Set label font size
    )
    
    # Show the Sankey diagram
    fig.show()




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


def get_data(data, level, number_category, type_classification="notspecified", Use_Others=True):
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
    

    data = data.loc[data['nb_niv'] >= 3].copy()

    # Extract the specified taxonomic level
    data.loc[:, 'Labels'] = data['Taxonomic'].apply(lambda x: extract_word_at_position(x, level - 1))
    
    if type_classification != "notspecified":
        data.loc[:, 'previous level'] = data['Taxonomic'].apply(lambda x: extract_word_at_position(x, level - 2))
        data.loc[:, 'previous level'] = data['previous level'].str.replace(' ', '')
        specify_data = data[data['previous level'] == type_classification].copy()
    else:
        specify_data = data.copy()
    
    specify_data.loc[:, f'Labels'] = specify_data['Labels'].str.replace(' ', '')

    if Use_Others == False:
        number_category+= 1

    name_category = specify_data['Labels'].value_counts().head(number_category - 1).index.tolist()
    specify_data.loc[:, 'Labels'] = specify_data['Labels'].apply(
        lambda x: x if x in name_category else 'Others')

    if Use_Others:
        name_category.append('Others')
    else: 
        specify_data.drop(specify_data[specify_data['Labels'] == 'Others'].index, inplace=True)


    specify_data = specify_data.drop(['nb_niv', 'Taxonomic', 'Organism', 'Entry'], axis=1)
    if type_classification != "notspecified":
        specify_data = specify_data.drop(['previous level'], axis=1)

    specify_data = specify_data.reset_index(drop=True)
    specify_data.loc[:, f'Labels'] = specify_data['Labels'].astype("category")
    
    for abc in name_category:
        print(f'number of {abc} : ')
        print({specify_data['Labels'].value_counts().get(abc, 0)})
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