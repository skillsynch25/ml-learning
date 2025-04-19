import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def preprocess_data(X, y=None, scaler=None):
    """
    Preprocess the data by handling missing values and scaling features
    """
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Scale features
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    if y is not None:
        y = y.fillna(y.mean())
        return X_scaled, y, scaler
    return X_scaled, scaler

def plot_correlation_matrix(data):
    """
    Plot correlation matrix for the dataset
    """
    plt.figure(figsize=(10, 8))
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    return plt.gcf()

def plot_feature_distributions(data):
    """
    Plot distributions of all features
    """
    fig = px.histogram(data, 
                      title='Feature Distributions',
                      template='plotly_white')
    return fig

def plot_prediction_vs_actual(y_test, y_pred):
    """
    Plot actual vs predicted values
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    return plt.gcf()

def plot_residuals(y_test, y_pred):
    """
    Plot residuals
    """
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    return plt.gcf()

def generate_model_summary(model, X_train, feature_names):
    """
    Generate a summary of the model's performance and characteristics
    """
    summary = {
        'model_type': type(model).__name__,
        'n_features': X_train.shape[1],
        'feature_names': feature_names
    }
    
    if hasattr(model, 'coef_'):
        summary['coefficients'] = dict(zip(feature_names, model.coef_))
    
    if hasattr(model, 'feature_importances_'):
        summary['feature_importance'] = dict(zip(feature_names, model.feature_importances_))
    
    return summary 