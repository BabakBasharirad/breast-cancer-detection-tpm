
# Breast Cancer Detection using RNA-Seq TPM Data

This repository contains code and resources for detecting breast cancer using RNA-Seq gene expression data quantified as Transcripts Per Million (TPM). The project focuses on preprocessing, dimensionality reduction, and applying machine learning algorithms to predict cancerous tissue samples.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Data Sources](#data-sources)
- [Usage](#usage)
  - [Preprocessing](#preprocessing)
  - [Training the Model](#training-the-model)
  - [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Breast cancer is one of the most common cancers worldwide. In this project, we leverage RNA-Seq data to classify cancerous and non-cancerous tissue samples using machine learning. The main goal is to build a reliable model that can aid in early detection of breast cancer based on gene expression levels.

## Installation

To run this project, you will need the following dependencies:

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- TensorFlow (if using neural networks)
- Matplotlib
- Seaborn

You can install the dependencies using pip:

```bash
pip install -r requirements.txt
```

## Data Sources

The RNA-Seq gene expression data is sourced from two primary datasets:
- [Genomic Data Commons (GDC)](https://gdc.cancer.gov/)
- [Genotype-Tissue Expression (GTEx)](https://gtexportal.org/)

Preprocessed data is stored in the repository, but you can download and preprocess new data using the scripts provided in the `data/` folder.

## Usage

### Preprocessing

To preprocess the RNA-Seq TPM data, use the following script:

```bash
python preprocessing.py
```

This script performs normalization and dimensionality reduction using techniques such as PCA.

### Training the Model

To train the model, run:

```bash
python train.py
```

This will initiate training using the preprocessed data. You can modify the model architecture or hyperparameters by editing the `train.py` script.

### Evaluation

To evaluate the model on a test dataset, use:

```bash
python evaluate.py
```

Results including accuracy, precision, recall, and F1-score will be output.

## Results

The results of the trained models, including performance metrics and visualizations, can be found in the `results/` folder. This project utilizes various machine learning models, such as:

- Logistic Regression
- Random Forest
- Support Vector Machines
- Neural Networks

You can find detailed performance comparisons in the `results/report.pdf`.

## Contributing

Feel free to fork this repository and submit pull requests. For major changes, please open an issue to discuss what you would like to contribute.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
