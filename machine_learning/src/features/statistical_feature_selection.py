"""
Module 2: Statistical Feature Selection
This module performs statistical feature selection on the provided dataset by selecting 
the top k features (Default is 1000) based on the ANOVA F-value between label/feature pairs.
"""
# Standard Library Imports
import os

# Third-party Imports
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

# Local Imports
from config_loader import load_config 

def remove_constant_features(features):
    """Remove constant features from the dataset."""
    selector = VarianceThreshold()
    return selector.fit_transform(features), selector.get_support(indices=True)

def statistical_feature_selection(features_file, labels_file, output_file, k=1000):
    """
    Performs statistical feature selection on the provided dataset by selecting 
    the top k features based on the ANOVA F-value between label/feature pairs.

    Parameters:
    features_file (str): Path to the CSV file containing feature data.
    labels_file (str): Path to the CSV file containing labels.
    output_file (str): Path to the output CSV file to save selected features.
    k (int): Number of top features to select.
    """
    # Load preprocessed features and labels
    features = pd.read_csv(features_file)
    labels = pd.read_csv(labels_file)

    # Handle NaN values by imputing with the mean
    imputer = SimpleImputer(strategy='mean')
    features_imputed = imputer.fit_transform(features)

    # Remove constant features
    features_imputed, selected_indices = remove_constant_features(features_imputed)
    feature_names = features.columns[selected_indices]

    # Perform feature selection
    selector = SelectKBest(f_classif, k=k)
    selected_features = selector.fit_transform(features_imputed, labels['label'])

    # Get selected feature names
    feature_names = feature_names[selector.get_support()]
    selected_features_df = pd.DataFrame(selected_features, columns=feature_names)
    
    # Save the selected features to the output file
    selected_features_df.to_csv(output_file, index=False)

    print("Statistical Feature Selection Complete")
    print("Shape of selected features:", selected_features_df.shape)

if __name__ == "__main__":

    # Folder configuration
    config = load_config()
    initerim_data_dir = config['paths']['data_interim_dir'] 

    features_file = os.path.join(initerim_data_dir, 'preprocessed_data_features.csv')
    labels_file = os.path.join(initerim_data_dir,'preprocessed_data_labels.csv')
    output_file = os.path.join(initerim_data_dir, 'selected_features.csv')
    
    statistical_feature_selection(features_file, labels_file, output_file)

