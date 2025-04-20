# Machine Learning Learning Platform

A comprehensive platform for learning and experimenting with machine learning models, featuring detailed explanations, model interpretability, and step-by-step execution processes.

## Author
skillSynch- skillsynch.ai@gmail.com

## Educational Features

- **Interactive Learning Environment**
  - Step-by-step model training visualization
  - Real-time parameter tuning and effect observation
  - Interactive model architecture diagrams
  - Detailed mathematical explanations of model internals

- **Comprehensive Model Explanations**
  - Detailed theory behind each algorithm
  - Mathematical foundations and equations
  - Visual representations of model architectures
  - Real-world application examples

- **Learning Resources**
  - Built-in tutorials for each model type
  - Code examples with detailed comments
  - Best practices and common pitfalls
  - Performance optimization guides

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

## Model Explanations

### Linear Regression
- **Theory**: Linear regression models the relationship between a dependent variable and one or more independent variables using a linear equation.
- **Key Concepts**:
  - Ordinary Least Squares (OLS) estimation
  - Coefficient interpretation
  - Assumptions (linearity, independence, homoscedasticity)
  - R² and adjusted R² metrics
- **Parameters**:
  - Fit Intercept: Whether to calculate the intercept for this model
  - Normalize: Whether to normalize the features

### Random Forest
- **Theory**: An ensemble learning method that constructs multiple decision trees and combines their predictions.
- **Key Concepts**:
  - Bootstrap aggregating (bagging)
  - Feature importance
  - Out-of-bag error estimation
  - Decision tree splitting criteria
- **Parameters**:
  - Number of Trees: Number of trees in the forest
  - Max Depth: Maximum depth of the tree

### XGBoost
- **Theory**: A gradient boosting framework that uses tree-based learning algorithms.
- **Key Concepts**:
  - Gradient boosting
  - Regularization
  - Feature importance
  - Early stopping
- **Parameters**:
  - Learning Rate: Step size shrinkage used to prevent overfitting
  - Number of Trees: Number of boosting rounds
  - Max Depth: Maximum tree depth for base learners

### LightGBM
- **Theory**: A gradient boosting framework that uses tree-based learning algorithms with leaf-wise tree growth.
- **Key Concepts**:
  - Gradient-based One-Side Sampling (GOSS)
  - Exclusive Feature Bundling (EFB)
  - Leaf-wise tree growth
  - Histogram-based algorithm
- **Parameters**:
  - Learning Rate: Step size shrinkage used to prevent overfitting
  - Number of Trees: Number of boosting rounds
  - Max Depth: Maximum tree depth for base learners

### CatBoost (Optional)
- **Theory**: A gradient boosting algorithm that handles categorical features automatically.
- **Key Concepts**:
  - Ordered boosting
  - Categorical feature handling
  - Symmetric trees
  - Feature combinations
- **Parameters**:
  - Learning Rate: Step size shrinkage used to prevent overfitting
  - Number of Trees: Number of boosting rounds
  - Max Depth: Maximum tree depth for base learners

## Model Interpretability

The platform provides several tools for understanding model behavior:

1. **SHAP Values**: Shows the contribution of each feature to the model's predictions
   - Global feature importance
   - Individual prediction explanations
   - Interaction effects
   - Feature dependence plots

2. **LIME Explanations**: Provides local interpretable model-agnostic explanations
   - Local feature importance
   - Decision boundary visualization
   - Counterfactual explanations
   - Model behavior analysis

3. **Feature Importance**: Visualizes the relative importance of each feature
   - Permutation importance
   - Mean decrease in impurity
   - Feature correlation analysis
   - Partial dependence plots

4. **Model Metrics**: Displays performance metrics like MSE, R2 score, and MAE
   - Training vs validation performance
   - Learning curves
   - Residual analysis
   - Cross-validation results

## Learning Resources

### Tutorials
- Getting Started with Machine Learning
- Understanding Model Selection
- Feature Engineering Best Practices
- Hyperparameter Tuning Guide
- Model Evaluation Techniques

### Code Examples
- Data preprocessing
- Model implementation
- Cross-validation
- Hyperparameter optimization
- Model deployment

### Best Practices
- Data cleaning and preprocessing
- Feature selection
- Model selection criteria
- Performance optimization
- Common pitfalls to avoid

## Contributing

Feel free to submit issues and enhancement requests! We welcome contributions in the form of:
- New model implementations
- Educational content
- Code improvements
- Documentation updates
- Bug reports and fixes

## License

This project is licensed under the MIT License - see the LICENSE file for details. 