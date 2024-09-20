"""



"""

# Standard Library Imports
import os
from datetime import datetime

# Third-party Imports
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Local Imports
from config_loader import load_config
from report import dual_print


def histogram(benign_data, cancer_data):

    # Assuming 'gtex_data' and 'gdc_data' are your datasets
    plt.hist(benign_data.values.flatten(), bins=20, alpha=0.5, label='Benign Samples')
    plt.hist(cancer_data.values.flatten(), bins=20, alpha=0.5, label='Cancer Samples')
    # Set the y-axis to a logarithmic scale
    plt.yscale('log')

    plt.xlabel('Gene Expression Level')
    plt.ylabel('Frequency (log scale)')
    plt.legend(loc='upper right')
    plt.title('Distribution of Gene Expression Levels')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()


def pca_scatter_plot(features, labels):
    # Assuming 'features' is your feature matrix and 'labels' are your labels
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    pca_df['label'] = labels['label'].apply(lambda x: 'cancer' if x == 1 else 'benign')

    sns.scatterplot(x='PC1', y='PC2', hue='label', data=pca_df)
    plt.title('Principal Component Analysis of Data')
    plt.show()


if __name__ == "__main__":

    # Folder configuration
    config = load_config()
    processed_data_dir = config['paths']['data_processed_dir'] 

# Histogram
    data_file = os.path.join(processed_data_dir, 'dataset.csv')
    data = pd.read_csv(data_file)

    benign_data = data[data['label'] == 0].drop(columns=['label'])
    cancer_data = data[data['label'] == 1].drop(columns=['label'])

    print(benign_data.min().min(), benign_data.max().max())
    print(cancer_data.min().min(), cancer_data.max().max())
    histogram(benign_data, cancer_data)


# HistogramPCA Scatter Plot
    feature_file = os.path.join(processed_data_dir, 'features.csv')
    features = pd.read_csv(feature_file)

    label_file = os.path.join(processed_data_dir, 'labels.csv')
    labels = pd.read_csv(label_file)

    pca_scatter_plot(features, labels)

