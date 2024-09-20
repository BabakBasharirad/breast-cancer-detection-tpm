"""
Module 4: Model Training and Evaluation
This module trains and evaluates a Random Forest Classification model for cancer detection.
"""

# Standard Library Imports
import os

# Third-party Imports
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Local Imports
from config_loader import load_config


def model_training_evaluation(features_file, labels_file, model_file):
    """
    Trains and evaluates a machine learning model for cancer detection.

    Parameters:
    features_file (str): Path to the CSV file containing reduced feature data.
    labels_file (str): Path to the CSV file containing labels.
    model_file (str): Path to the output file to save the trained model.
    """
    # Load reduced features and labels
    features = pd.read_csv(features_file)
    labels = pd.read_csv(labels_file)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels['label'], test_size=0.2) #, random_state=42

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print evaluation metrics
    print("Model Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Save the model
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

    print(f"Trained model saved to {model_file}")

    # Assuming 'y_test' are true labels and 'y_pred' are predicted labels
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == "__main__":

    # Folder configuration
    config = load_config()
    processed_data_dir = config['paths']['data_processed_dir'] 
    model_dir = config['paths']['model_dir']

    features_file = os.path.join(processed_data_dir, 'features.csv')
    labels_file = os.path.join(processed_data_dir, 'labels.csv')
    model_file = os.path.join(model_dir, 'rfc_model.pkl')
    
    model_training_evaluation(features_file, labels_file, model_file)


