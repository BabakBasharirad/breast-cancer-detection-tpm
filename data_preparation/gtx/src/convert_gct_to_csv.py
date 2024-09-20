"""
This module process raw sample file downloaded from GTex portal.
The raw file is in format of *.gct. 
It reads the data and convert it to a csv format. 
The generated csv file will be then temporarily stored in the current folder for sampling. 
The temporary file contains a large amount of data, while will be processed further to 
randomly choose only 1000 samples. 'sampler.py' will do the next step.
"""

# Standard Library Imports
import os

# Third-party Imports
import pandas as pd

# Local Imports
from config_loader import load_config 

def read_gct(file_path):
    with open(file_path, 'r') as f:
        # Skip the first two header lines
        for _ in range(2):
            next(f)
        # Read the rest of the file into a pandas DataFrame
        df = pd.read_csv(f, sep='\t')
    return df

def convert_gct_to_csv(gct_file, csv_file):
    df = read_gct(gct_file)
    df.to_csv(csv_file, index=False)

# Folder configuration
config = load_config()

gdc_raw_data_files_sub_dir = config['paths']['raw_gtx_data_files_dir']
gtex_gct_file = config['files']['gtx_raw_data_file']
gtex_gct_file_full = os.path.join(gdc_raw_data_files_sub_dir, gtex_gct_file)

# Path to the output .csv file
gtx_df_file = os.path.join(gdc_raw_data_files_sub_dir, 'gtx_raw_data.csv')
convert_gct_to_csv(gtex_gct_file_full, gtx_df_file)

first_rows = gtx_df_file.head(15) 
print(first_rows)