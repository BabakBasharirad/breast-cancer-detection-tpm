"""
model_cross_validation.py
This module performs cross-validation on the trained machine learning model.
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
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Local Imports
from config_loader import load_config
from report import dual_print


def cross_validate_model(features_file, labels_file, model_file_prefix, report_file, num_splits=5):
    """
    Performs cross-validation on the trained machine learning model.

    Parameters:
    reduced_features_file (str): Path to the CSV file containing reduced feature data.
    labels_file (str): Path to the CSV file containing labels.
    n_splits (int): Number of cross-validation splits. Default is 5.
    """

    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")

    report_f = open(report_file, 'w')
    dual_print(report_file, 'This report is generated on ', formatted_now, '\n')
    report_f.close()

    # Load reduced features and labels
    features = pd.read_csv(features_file)
    labels = pd.read_csv(labels_file)

    # Perform cross-validation
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)

    # Method 1
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    fig = plt.figure()
    fig.suptitle('Confusion Matrix', fontsize=12, fontweight="bold")

    gs = gridspec.GridSpec(2, 6)

    # Perform cross-validation
    model_number = 0
    for train_index, test_index in skf.split(features, labels):

        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = labels['label'].iloc[train_index], labels['label'].iloc[test_index]

        model_number += 1

        # Train the model
        model = RandomForestClassifier(n_estimators=100) #, random_state=42
        model.fit(X_train, y_train)        
        
        # Save the model
        model_file = model_file_prefix + str(model_number) + '.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)

        dual_print(report_file, f"Trained model saved to {model_file}")

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        dual_print(report_file, f'Model{model_number}: Training Samples={len(y_train)} Training Samples={len(y_test)}\
              TP={tp}, FN={fn}, TN={tn}, FP={fp}\n')

        i = model_number - 1
        if i < 3:
            ax = plt.subplot(gs[0, 2 * i:2 * i + 2])
        else:
            ax = plt.subplot(gs[1, 2 * i - 5:2 * i + 2 - 5])

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(f'Fold {model_number}', fontsize=10, fontweight="bold")
        ax.set_xlabel('Prediction', fontsize = 8, fontweight="bold")
        ax.set_ylabel('Actual', fontsize = 8, fontweight="bold")
        ax.set_xticklabels(['Benign', 'Cancer'], fontsize=6, rotation=60)
        ax.set_yticklabels(['Benign', 'Cancer'], fontsize=6, rotation=60)

    # Print average metrics
    dual_print(report_file, "Cross-Validation Results:")
    dual_print(report_file, f"Accuracy: {sum(accuracy_scores) / len(accuracy_scores):.4f}")
    dual_print(report_file, f"Precision: {sum(precision_scores) / len(precision_scores):.4f}")
    dual_print(report_file, f"Recall: {sum(recall_scores) / len(recall_scores):.4f}")
    dual_print(report_file, f"F1 Score: {sum(f1_scores) / len(f1_scores):.4f}")

    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=1.5, wspace=0.4)
    plt.tight_layout(pad=2)
    plt.show()

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
    cross_validate_model(features_file, labels_file, model_file_prefix, report_file)
