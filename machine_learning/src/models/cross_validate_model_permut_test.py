"""
model_cross_validation.py
This module performs cross-validation on the trained machine learning model
and includes a permutation test for additional validation.

Parameters:
    features_file (str): Path to the CSV file containing reduced feature data.
    labels_file (str): Path to the CSV file containing labels.
"""

# Standard Library Imports
import os
from datetime import datetime

# Third-party Imports
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Local Imports
from config_loader import load_config
from report import dual_print

def cross_validate_model(features_file, labels_file, model_file_prefix, report_file, num_splits=5, n_permutations=1000):
    """
    Performs cross-validation on the trained machine learning model
    and conducts a permutation test to assess model robustness.

    Parameters:
        features_file (str): Path to CSV file containing features.
        labels_file (str): Path to CSV file containing labels.
        model_file_prefix (str): Prefix for saving the trained model.
        report_file (str): Path to the report file for logging results.
        num_splits (int): Number of folds for Stratified K-Fold CV (default: 5).
        n_permutations (int): Number of label permutations for the permutation test (default: 1000).
    """

    # Load data
    X = pd.read_csv(features_file)
    y = pd.read_csv(labels_file).values.ravel()

    # Initialize classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    accuracies, precisions, recalls, f1_scores = [], [], [], []
    confusion_matrices = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, zero_division=1))
        recalls.append(recall_score(y_test, y_pred, zero_division=1))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=1))

        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices.append(cm)

    # Compute averages
    avg_accuracy = sum(accuracies) / num_splits
    avg_precision = sum(precisions) / num_splits
    avg_recall = sum(recalls) / num_splits
    avg_f1_score = sum(f1_scores) / num_splits

    # Log Cross-Validation Results
    dual_print(report_file, f"Cross-Validation Results:")
    dual_print(report_file, f"  Accuracy: {avg_accuracy:.4f}")
    dual_print(report_file, f"  Precision: {avg_precision:.4f}")
    dual_print(report_file, f"  Recall: {avg_recall:.4f}")
    dual_print(report_file, f"  F1 Score: {avg_f1_score:.4f}")

    # Perform Permutation Test
    score, permutation_scores, p_value = permutation_test_score(
        model, X, y, scoring="accuracy", cv=skf, n_permutations=n_permutations, random_state=42
    )

    # Log Permutation Test Results
    dual_print(report_file, f"Permutation Test Results:")
    dual_print(report_file, f"  Original Model Accuracy: {score:.4f}")
    dual_print(report_file, f"  Mean Permuted Accuracy: {permutation_scores.mean():.4f}")
    dual_print(report_file, f"  p-value: {p_value:.4f}")

    if p_value < 0.05:
        dual_print(report_file, "  The model significantly outperforms random chance (p < 0.05).")
    else:
        dual_print(report_file, "  The model's performance is not significantly better than random.")

    # Save final trained model
    with open(f"{model_file_prefix}_random_forest.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

    # Plot Confusion Matrix
    fig, axes = plt.subplots(1, num_splits, figsize=(15, 5))
    for i, cm in enumerate(confusion_matrices):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=axes[i])
        axes[i].set_title(f"Fold {i+1}")

    plt.tight_layout()
    plt.savefig(f"{model_file_prefix}_confusion_matrix.png")
    plt.close()

if __name__ == "__main__":

    # Folder configuration
    config = load_config()
    processed_data_dir = config['paths']['data_processed_dir'] 
    model_dir = config['paths']['model_dir']
    report_dir = config['paths']['report_dir']

    features_file = os.path.join(processed_data_dir, 'features.csv')
    labels_file = os.path.join(processed_data_dir, 'labels.csv')
    model_file_prefix = os.path.join(model_dir, 'rfc_model_5F_CV_')
    report_file = os.path.join(report_dir, 'CV_report.txt')

    cross_validate_model(features_file, labels_file, model_file_prefix, report_file,
        num_splits=5, n_permutations=100)