import pandas as pd
import shap
import pickle

# Load the model
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the data
features = pd.read_csv('reduced_features.csv')

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(features)

# Plot SHAP values for a sample
shap.summary_plot(shap_values, features)
