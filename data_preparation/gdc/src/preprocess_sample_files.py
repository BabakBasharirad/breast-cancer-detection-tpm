""" 
This module process the sample files of Cancers downloaded from GDC portal
A sample sheet (downloaded from GDC portal) is used to locate sample files (*.tsv) 
in sub-directory
The column "tpm_unstranded" is extracted from each file as the main feature
These data are then merged to create a file "gdc_data.csv", structured as below:
Columns present samples, and rows present genes. Each data shows Transcript per Milion (TPM)
of a sapecific gene in a sample.

The generated csv file will be then stored in (initial) data sub-direcotory in machine learning folder, for
further preprocessing phase.
"""

# Standard Library Imports
import os

# Third-party Imports
import pandas as pd

# Local Imports
from config_loader import load_config 

# Folder configuration
config = load_config()
gdc_raw_data_sub_dir = config['paths']['raw_gdc_data_dir']
gdc_raw_data_files_sub_dir = config['paths']['raw_gdc_data_files_dir']

# Load the sample sheet
# Sample sheet file is downloaded from the GDC website.
# It contains information about the samples file, including file ID, file name, sample ID, etc.
# This file is used to locate the sample files
sample_sheet_file_name = config['files']['gdc_sample_sheet_file']
sample_sheet_file_full_path = os.path.join(gdc_raw_data_sub_dir, sample_sheet_file_name)

sample_sheet_file_handler = pd.read_csv(sample_sheet_file_full_path, sep='\t')

# Initialize an empty DataFrame to store the combined data
gdc_df = pd.DataFrame()

sample_counter = 1
# Iterate through each file listed in the sample sheet
for index, row in sample_sheet_file_handler.iterrows():
    folder_id = row['File ID']
    file_name = row['File Name']
    
    # Construct the full file path
    file_path = os.path.join(gdc_raw_data_files_sub_dir, folder_id, file_name)
    
    # Read the TSV file
    try:
        sample_data = pd.read_csv(file_path, sep='\t', skiprows=[0, 2, 3, 4, 5])
        
        # Extract the relevant columns ('gene_id' and 'TPM' or equivalent)
        relevant_data = sample_data[['gene_id', 'tpm_unstranded']]
        
        # Rename the columns to match the GTEx format
        sample_number = 'sample_' + str(sample_counter).zfill(4)
        relevant_data.columns = ['gene_id', sample_number]  # Use folder_id as the sample name
        
        # Merge with the combined data
        if gdc_df.empty:
            gdc_df = relevant_data
        else:
            gdc_df = pd.merge(gdc_df, relevant_data, on='gene_id', how='outer')

        sample_counter += 1
            
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# # Create labels: 1 for GDC (Cancer)
# new_row = pd.DataFrame([[1] * len(gdc_df.columns)], columns=gdc_df.columns)
# new_row.iloc[0, 0] = 'Label'

# # Add the labels as the first row
# gdc_df_with_label = pd.concat([new_row, gdc_df], ignore_index=False)

# # Reset the index to reflect the new row properly
# gdc_df_with_label.index = ['Label'] + list(gdc_df.index)

# Save the merged data to a CSV file
initial_data_dir = config['paths']['data_initial_dir']

gdc_df_file = os.path.join(initial_data_dir, 'gdc_data.csv')
gdc_df.to_csv(gdc_df_file, index=False)

# Display the first few rows of the combined data
print(gdc_df.head())