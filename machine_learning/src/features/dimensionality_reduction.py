"""
Module 3: Dimensionality Reduction.py
This module performs dimensionality reduction on the provided dataset using PCA.
"""
# Standard Library Imports
import os

# Third-party Imports
import pandas as pd
from sklearn.decomposition import PCA

# Local Imports
from config_loader import load_config


def dimensionality_reduction(selected_features_file, output_file, n_components=50):
    """
    Performs dimensionality reduction on the provided dataset using PCA.

    Parameters:
    selected_features_file (str): Path to the CSV file containing selected feature data.
    output_file (str): Path to the output CSV file to save reduced features.
    n_components (int): Number of principal components to reduce to. Default is 50.
    """
    # Load selected features
    selected_features = pd.read_csv(selected_features_file)

    # Perform PCA
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(selected_features)

    # Create a DataFrame with reduced features
    reduced_features_df = pd.DataFrame(reduced_features, columns=[f'PC{i+1}' for i in range(n_components)])
    
    # Save the reduced features to the output file
    reduced_features_df.to_csv(output_file, index=False)

    print("Dimensionality Reduction Complete")
    print("Shape of reduced features:", reduced_features_df.shape)


if __name__ == "__main__":

    # Folder configuration
    config = load_config()
    initerim_data_dir = config['paths']['data_interim_dir']
    processed_data_dir = config['paths']['data_processed_dir']

    labels_file = config['files']['lables_file']
    features_file = config['files']['features_file']
    dataset_file = config['files']['dataset_file']
    benign_dataset_file = config['files']['benign_dataset_file']
    cancer_dataset_file = config['files']['cancer_dataset_file']


    selected_features_file = os.path.join(initerim_data_dir, 'selected_features.csv')
    output_file = os.path.join(processed_data_dir, 'features.csv')
    
    n_components = 50
    
    dimensionality_reduction(selected_features_file, output_file, n_components)
    
    labels_df = pd.read_csv(os.path.join(processed_data_dir, labels_file))
    features_df = pd.read_csv(os.path.join(processed_data_dir, features_file))
   
    dataset_df = pd.concat([features_df, labels_df], axis=1)
    dataset_df.to_csv(os.path.join(processed_data_dir, dataset_file), index=False)

    benign_dataset = dataset_df[dataset_df['label'] == 0]
    benign_dataset.to_csv(os.path.join(processed_data_dir, benign_dataset_file), index=False)

    cancer_dataset = dataset_df[dataset_df['label'] == 1]
    cancer_dataset.to_csv(os.path.join(processed_data_dir, cancer_dataset_file), index=False)
