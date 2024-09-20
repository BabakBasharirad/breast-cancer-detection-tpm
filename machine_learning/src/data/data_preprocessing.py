"""
Step 1: Data Preprocessing
This module preprocesses the GTEx and GDC datasets by merging them, adding labels, and 
transposing the data to have samples as rows and genes as columns.
"""    

# Standard Library Imports
import os

# Third-party Imports
import pandas as pd

# Local Imports
from config_loader import load_config 

def preprocess_data(gtx_file, gdc_file, output_file_prefix):
    """
    Preprocesses the GTEx and GDC datasets by merging them, adding labels, and 
    transposing the data to have samples as rows and genes as columns.

    Parameters:
    gtx_file (str): Path to the GTEx dataset file (CSV).
    gdc_file (str): Path to the GDC dataset file (CSV).
    output_file (str): Prefix for the output files.

    Returns:
    None: The function saves two CSV files, one for features and one for labels.
    """    
    
    # Load the datasets with headers
    gtx_data = pd.read_csv(gtx_file, header=0)
    gdc_data = pd.read_csv(gdc_file, header=0)

    # Ensure that the 'gene_id' column is the index for both datasets
    gtx_data.set_index('gene_id', inplace=True)
    gdc_data.set_index('gene_id', inplace=True)

    # Add labels row: 0 for GTEx (healthy) and 1 for GDC (cancer)
    gtx_labels = pd.DataFrame([0] * gtx_data.shape[1], index=gtx_data.columns, columns=['label']).transpose()
    gdc_labels = pd.DataFrame([1] * gdc_data.shape[1], index=gdc_data.columns, columns=['label']).transpose()

    # Concatenate labels and data
    gtx_data = pd.concat([gtx_labels, gtx_data])
    gdc_data = pd.concat([gdc_labels, gdc_data])

    # Combine both datasets
    combined_data = pd.concat([gtx_data, gdc_data], axis=1)

    # Transpose the data to have samples as rows and genes as columns
    combined_data = combined_data.transpose()

    # Separate features and labels
    labels = combined_data['label']
    features = combined_data.drop(columns=['label'])

    # Save preprocessed features and labels to CSV
    features.to_csv(output_file_prefix + '_features.csv', index=False)
    labels.to_csv(output_file_prefix + '_labels.csv', index=False)

    print("Data Preprocessing Complete")
    print("Shape of features:", features.shape)
    print("Shape of labels:", labels.shape)

if __name__ == "__main__":

    # Folder configuration
    config = load_config()
    initial_data_dir = config['paths']['data_initial_dir']
    initerim_data_dir = config['paths']['data_interim_dir'] 
    processed_data_dir = config['paths']['data_processed_dir'] 

    gdc_data_file = os.path.join(initial_data_dir, 'gdc_data.csv')
    gtx_data_file = os.path.join(initial_data_dir, 'gtx_data.csv')
    output_file_prefix = os.path.join(initerim_data_dir, 'preprocessed_data')
    
    label_file_in_interim = os.path.join(initerim_data_dir, 'preprocessed_data_labels.csv')
    label_file_in_processed = os.path.join(processed_data_dir, 'labels.csv')

    import shutil
    shutil.copyfile(label_file_in_interim, label_file_in_processed)

    preprocess_data(gtx_data_file, gdc_data_file, output_file_prefix)
