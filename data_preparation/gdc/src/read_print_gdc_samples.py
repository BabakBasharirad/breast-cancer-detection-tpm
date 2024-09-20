""" 
This module reads initial processed data file which is ready for the model,
and pront first few rows, just to check.

This modeuls is used to print the first 15 rows of sample data file, to be placed in the paper.
"""

# Standard Library Imports
import os

# Third-party Imports
import pandas as pd

# Local Imports
from config_loader import load_config 

# Folder configuration
config = load_config()
initial_data_dir = config['paths']['data_initial_dir']
gdc_samples_file = os.path.join(initial_data_dir, 'gdc_data.csv')

# Load the sample sheet
gdc_samples_df = pd.read_csv(gdc_samples_file, sep=',')

# Display the first few rows of the combined data

first_rows = gdc_samples_df.head(15) 
print(first_rows)