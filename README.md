# Machine Learning Learning Platform

A comprehensive platform for learning and experimenting with machine learning models, featuring detailed explanations, model interpretability, and step-by-step execution processes.

## Features

- Multiple ML models support:
  - Linear Regression
  - Random Forest
  - XGBoost
  - LightGBM
  - CatBoost (optional, see installation notes)
- Interactive parameter tuning
- Model interpretability tools:
  - SHAP values
  - LIME explanations
  - Feature importance analysis
- Step-by-step execution process
- Detailed model metrics and visualizations

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Optional: CatBoost Installation
CatBoost is an optional dependency that may require additional setup on Windows. To install it:
```bash
pip install catboost==1.2.5
```
Note: If you encounter installation issues with CatBoost, you can still use all other models in the platform.

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Upload your dataset (CSV format)
3. Select the target column and feature columns
4. Choose a model type and adjust its parameters
5. Click "Train Model" to start the training process
6. Explore the results, including:
   - Model performance metrics
   - SHAP values
   - LIME explanations
   - Feature importance
   - Step-by-step execution process

## Model Parameters

### Linear Regression
- Fit Intercept: Whether to calculate the intercept for this model
- Normalize: Whether to normalize the features

### Random Forest
- Number of Trees: Number of trees in the forest
- Max Depth: Maximum depth of the tree

### XGBoost
- Learning Rate: Step size shrinkage used to prevent overfitting
- Number of Trees: Number of boosting rounds
- Max Depth: Maximum tree depth for base learners

### LightGBM
- Learning Rate: Step size shrinkage used to prevent overfitting
- Number of Trees: Number of boosting rounds
- Max Depth: Maximum tree depth for base learners

### CatBoost (Optional)
- Learning Rate: Step size shrinkage used to prevent overfitting
- Number of Trees: Number of boosting rounds
- Max Depth: Maximum tree depth for base learners

## Model Interpretability

The platform provides several tools for understanding model behavior:

1. **SHAP Values**: Shows the contribution of each feature to the model's predictions
2. **LIME Explanations**: Provides local interpretable model-agnostic explanations
3. **Feature Importance**: Visualizes the relative importance of each feature
4. **Model Metrics**: Displays performance metrics like MSE, R2 score, and MAE

## Contributing

Feel free to submit issues and enhancement requests! 