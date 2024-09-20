"""
This sample will choolse 1000 random samples from the gtex data file.
The generated csv file will be then stored in (initial) data sub-direcotory in machine learning folder, for
further preprocessing phase.
"""

# Standard Library Imports
import os

# Third-party Imports
import pandas as pd
import numpy as np

# Local Imports
from config_loader import load_config 

# Number of sample columns to extract
num_samples = 1000

# Folder configuration
config = load_config()
initial_data_dir = config['paths']['raw_gtx_data_files_dir']
gtx_big_samples_file = os.path.join(initial_data_dir, 'gtx_raw_data.csv')

# Load the GTEx data with only the first few rows to inspect the column names
gtex_data = pd.read_csv(gtx_big_samples_file, nrows=5)
columns = gtex_data.columns.tolist()

# The first two columns are 'Name' and 'Description', we keep them and sample the rest
gene_info_columns = columns[0:1]
sample_columns = columns[2:]

# Randomly select 1000 sample columns from the dataset
sampled_columns = np.random.choice(sample_columns, num_samples, replace=False).tolist()

# Combine gene info columns with the sampled columns
final_columns = gene_info_columns + sampled_columns

# Load the dataset again but only with the selected columns
gtex_df = pd.read_csv(gtx_big_samples_file, usecols=final_columns)
gtex_df = gtex_df.rename(columns={'Name': 'gene_id'})

# Save the sampled data to a new CSV file
# Path to the output .csv file
initial_data_dir = config['paths']['data_initial_dir']
gtx_df_file = os.path.join(initial_data_dir, 'gtx_data.csv')

gtex_df.to_csv(gtx_df_file, index=False)

first_rows = gtex_df.head(15) 
print(first_rows)