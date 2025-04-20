import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns

class LinearRegressionModel:
    """
    Linear Regression Model with detailed educational content and visualizations.
    
    Educational Notes:
    - Linear Regression is one of the simplest and most widely used statistical techniques
    - It models the relationship between a dependent variable and one or more independent variables
    - The model assumes a linear relationship between the input features and the target variable
    - The goal is to find the best-fitting straight line through the data points
    
    Key Concepts:
    1. Hypothesis Function: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
    2. Cost Function: Mean Squared Error (MSE)
    3. Optimization: Gradient Descent or Normal Equation
    4. Assumptions: Linearity, Independence, Homoscedasticity, Normality
    
    Visualizations:
    - Scatter plots with regression line
    - Residual plots
    - Coefficient importance
    """
    
    def __init__(self, fit_intercept=True, normalize=False):
        self.model = LinearRegression(fit_intercept=fit_intercept)
        self.scaler = StandardScaler() if normalize else None
        self.coefficients = None
        self.intercept = None
        
    def fit(self, X, y):
        """
        Fit the linear regression model to the training data.
        
        Educational Notes:
        - The fit method estimates the model parameters (coefficients)
        - If normalize=True, features are standardized to have zero mean and unit variance
        - The coefficients represent the change in the target variable for a one-unit change in the feature
        """
        if self.scaler:
            X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        self.coefficients = self.model.coef_
        self.intercept = self.model.intercept_
        
    def predict(self, X):
        """Make predictions using the fitted model."""
        if self.scaler:
            X = self.scaler.transform(X)
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """
        Evaluate the model performance.
        
        Educational Notes:
        - R² score: Proportion of variance in the target variable explained by the model
        - MSE: Average squared difference between predicted and actual values
        - RMSE: Square root of MSE, in the same units as the target variable
        """
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2
        }
    
    def plot_regression_line(self, X, y, feature_name):
        """
        Plot the regression line for a single feature.
        
        Educational Notes:
        - Visualizes the relationship between a feature and the target variable
        - The slope of the line represents the coefficient
        - Points above/below the line are over/under predictions
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, alpha=0.5)
        plt.plot(X, self.predict(X.reshape(-1, 1)), color='red')
        plt.xlabel(feature_name)
        plt.ylabel('Target')
        plt.title('Linear Regression Fit')
        plt.show()
    
    def plot_residuals(self, X, y):
        """
        Plot the residuals to check model assumptions.
        
        Educational Notes:
        - Residuals should be randomly distributed around zero
        - No clear patterns indicate good model fit
        - Heteroscedasticity (changing variance) is a sign of model misspecification
        """
        y_pred = self.predict(X)
        residuals = y - y_pred
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.show()

class RandomForestModel:
    """
    Random Forest Model with detailed educational content and visualizations.
    
    Educational Notes:
    - Random Forest is an ensemble learning method that combines multiple decision trees
    - It uses bagging (bootstrap aggregating) and random feature selection
    - Each tree is trained on a random subset of the data and features
    - The final prediction is the average (regression) or majority vote (classification)
    
    Key Concepts:
    1. Decision Trees: Base learners that recursively split the data
    2. Bagging: Training multiple models on different data samples
    3. Feature Importance: Measures the contribution of each feature
    4. Out-of-Bag Error: Internal validation using unused samples
    
    Visualizations:
    - Feature importance plots
    - Decision tree structure
    - Partial dependence plots
    """
    
    def __init__(self, task='regression', n_estimators=100, max_depth=None):
        """
        Initialize the Random Forest model.
        
        Parameters:
        - task: 'regression' or 'classification'
        - n_estimators: Number of trees in the forest
        - max_depth: Maximum depth of each tree
        
        Educational Notes:
        - More trees generally lead to better performance but slower training
        - Deeper trees can capture more complex patterns but may overfit
        - The optimal parameters depend on the dataset and problem
        """
        if task == 'regression':
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
        self.task = task
        
    def fit(self, X, y):
        """Fit the random forest model to the training data."""
        self.model.fit(X, y)
        
    def predict(self, X):
        """Make predictions using the fitted model."""
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """
        Evaluate the model performance.
        
        Educational Notes:
        - For regression: MSE, RMSE, R²
        - For classification: Accuracy, Precision, Recall, F1-score
        - Feature importance shows which features contribute most to predictions
        """
        y_pred = self.predict(X)
        
        if self.task == 'regression':
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            return {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2
            }
        else:
            accuracy = accuracy_score(y, y_pred)
            cm = confusion_matrix(y, y_pred)
            return {
                'accuracy': accuracy,
                'confusion_matrix': cm
            }
    
    def plot_feature_importance(self, feature_names):
        """
        Plot the importance of each feature.
        
        Educational Notes:
        - Feature importance measures how much each feature contributes to predictions
        - Higher values indicate more important features
        - Can help in feature selection and understanding the problem
        """
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), feature_names[indices], rotation=45)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()

class SVMModel:
    """
    Support Vector Machine Model with detailed educational content and visualizations.
    
    Educational Notes:
    - SVM finds the optimal hyperplane that maximizes the margin between classes
    - It can handle both linear and non-linear problems using kernel functions
    - The model is robust to outliers and works well in high-dimensional spaces
    - Support vectors are the data points that define the decision boundary
    
    Key Concepts:
    1. Margin: Distance between the decision boundary and the nearest points
    2. Kernel Trick: Maps data to higher dimensions without explicit transformation
    3. Regularization: Controls the trade-off between margin width and training error
    4. Support Vectors: Critical points that define the decision boundary
    
    Visualizations:
    - Decision boundary plots
    - Support vectors visualization
    - Kernel space transformation
    """
    
    def __init__(self, task='regression', kernel='rbf', C=1.0, gamma='scale'):
        """
        Initialize the SVM model.
        
        Parameters:
        - task: 'regression' or 'classification'
        - kernel: Type of kernel function ('linear', 'rbf', 'poly', 'sigmoid')
        - C: Regularization parameter
        - gamma: Kernel coefficient
        
        Educational Notes:
        - C controls the trade-off between margin width and training error
        - Kernel choice depends on the data structure and problem
        - Gamma affects the influence of each training example
        """
        if task == 'regression':
            self.model = SVR(kernel=kernel, C=C, gamma=gamma)
        else:
            self.model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
        self.task = task
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        """
        Fit the SVM model to the training data.
        
        Educational Notes:
        - Features are standardized to have zero mean and unit variance
        - The model finds the optimal hyperplane in the transformed space
        - Support vectors are stored for prediction and visualization
        """
        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        
    def predict(self, X):
        """Make predictions using the fitted model."""
        X = self.scaler.transform(X)
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get probability estimates for classification tasks."""
        if self.task == 'classification':
            X = self.scaler.transform(X)
            return self.model.predict_proba(X)
        return None
    
    def evaluate(self, X, y):
        """
        Evaluate the model performance.
        
        Educational Notes:
        - For regression: MSE, RMSE, R²
        - For classification: Accuracy, Precision, Recall, F1-score
        - Support vectors can be used to understand the decision boundary
        """
        y_pred = self.predict(X)
        
        if self.task == 'regression':
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            return {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2
            }
        else:
            accuracy = accuracy_score(y, y_pred)
            cm = confusion_matrix(y, y_pred)
            return {
                'accuracy': accuracy,
                'confusion_matrix': cm
            }
    
    def plot_decision_boundary(self, X, y, feature_names):
        """
        Plot the decision boundary for 2D data.
        
        Educational Notes:
        - Visualizes how the model separates different classes
        - Support vectors are highlighted
        - The margin is shown as a shaded region
        """
        if len(feature_names) != 2:
            print("Decision boundary plot requires exactly 2 features")
            return
            
        X = self.scaler.transform(X)
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))
        
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.figure(figsize=(10, 6))
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title('SVM Decision Boundary')
        plt.show()

class GradientBoostingModel:
    """
    Gradient Boosting Model with detailed educational content and visualizations.
    
    Educational Notes:
    - Gradient Boosting builds an ensemble of weak learners sequentially
    - Each new model corrects the errors of the previous ones
    - It can handle both regression and classification tasks
    - The model is robust to outliers and can capture complex patterns
    
    Key Concepts:
    1. Weak Learners: Simple models (usually decision trees) that perform slightly better than random
    2. Gradient Descent: Minimizes the loss function by iteratively adding models
    3. Learning Rate: Controls the contribution of each tree
    4. Feature Importance: Measures the contribution of each feature
    
    Visualizations:
    - Learning curves
    - Feature importance plots
    - Partial dependence plots
    """
    
    def __init__(self, task='regression', model_type='xgboost', learning_rate=0.1, n_estimators=100):
        """
        Initialize the Gradient Boosting model.
        
        Parameters:
        - task: 'regression' or 'classification'
        - model_type: 'xgboost', 'lightgbm', or 'catboost'
        - learning_rate: Step size for each iteration
        - n_estimators: Number of boosting rounds
        
        Educational Notes:
        - Different implementations have different strengths
        - Learning rate controls the contribution of each tree
        - More estimators generally lead to better performance
        """
        self.task = task
        self.model_type = model_type
        
        if model_type == 'xgboost':
            if task == 'regression':
                self.model = xgb.XGBRegressor(
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    random_state=42
                )
            else:
                self.model = xgb.XGBClassifier(
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    random_state=42
                )
        elif model_type == 'lightgbm':
            if task == 'regression':
                self.model = lgb.LGBMRegressor(
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    random_state=42
                )
            else:
                self.model = lgb.LGBMClassifier(
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    random_state=42
                )
        else:  # catboost
            if task == 'regression':
                self.model = CatBoostRegressor(
                    learning_rate=learning_rate,
                    iterations=n_estimators,
                    random_seed=42,
                    verbose=False
                )
            else:
                self.model = CatBoostClassifier(
                    learning_rate=learning_rate,
                    iterations=n_estimators,
                    random_seed=42,
                    verbose=False
                )
        
    def fit(self, X, y):
        """
        Fit the gradient boosting model to the training data.
        
        Educational Notes:
        - The model builds trees sequentially, each correcting the previous errors
        - Early stopping can prevent overfitting
        - Feature importance is calculated during training
        """
        self.model.fit(X, y)
        
    def predict(self, X):
        """Make predictions using the fitted model."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get probability estimates for classification tasks."""
        if self.task == 'classification':
            return self.model.predict_proba(X)
        return None
    
    def evaluate(self, X, y):
        """
        Evaluate the model performance.
        
        Educational Notes:
        - For regression: MSE, RMSE, R²
        - For classification: Accuracy, Precision, Recall, F1-score
        - Learning curves show how the model improves with each iteration
        """
        y_pred = self.predict(X)
        
        if self.task == 'regression':
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            return {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2
            }
        else:
            accuracy = accuracy_score(y, y_pred)
            cm = confusion_matrix(y, y_pred)
            return {
                'accuracy': accuracy,
                'confusion_matrix': cm
            }
    
    def plot_learning_curves(self, X_train, y_train, X_val, y_val):
        """
        Plot the learning curves showing training and validation performance.
        
        Educational Notes:
        - Shows how the model's performance improves with each iteration
        - Helps identify overfitting (increasing gap between curves)
        - Can guide the choice of optimal number of iterations
        """
        train_scores = []
        val_scores = []
        
        for i in range(1, self.model.n_estimators + 1):
            self.model.n_estimators = i
            self.model.fit(X_train, y_train)
            
            train_pred = self.model.predict(X_train)
            val_pred = self.model.predict(X_val)
            
            if self.task == 'regression':
                train_scores.append(r2_score(y_train, train_pred))
                val_scores.append(r2_score(y_val, val_pred))
            else:
                train_scores.append(accuracy_score(y_train, train_pred))
                val_scores.append(accuracy_score(y_val, val_pred))
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_scores) + 1), train_scores, label='Training')
        plt.plot(range(1, len(val_scores) + 1), val_scores, label='Validation')
        plt.xlabel('Number of Trees')
        plt.ylabel('Score')
        plt.title('Learning Curves')
        plt.legend()
        plt.show()
    
    def plot_feature_importance(self, feature_names):
        """
        Plot the importance of each feature.
        
        Educational Notes:
        - Shows which features contribute most to predictions
        - Can help in feature selection and understanding the problem
        - Different implementations may calculate importance differently
        """
        if self.model_type == 'xgboost':
            importance = self.model.feature_importances_
        elif self.model_type == 'lightgbm':
            importance = self.model.feature_importances_
        else:  # catboost
            importance = self.model.get_feature_importance()
        
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), feature_names[indices], rotation=45)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show() 