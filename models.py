from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor

def train_linear_regression(X_train, y_train, fit_intercept=True, normalize=False):
    """
    Train a Linear Regression model
    """
    # Note: normalize parameter is deprecated, use StandardScaler instead
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
    
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1):
    """
    Train a Random Forest model
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, learning_rate=0.1, n_estimators=100, max_depth=6, subsample=1.0, colsample_bytree=1.0):
    """
    Train an XGBoost model
    """
    model = xgb.XGBRegressor(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_lightgbm(X_train, y_train, learning_rate=0.1, n_estimators=100, max_depth=6, num_leaves=31, subsample=1.0):
    """
    Train a LightGBM model
    
    Parameters:
    -----------
    X_train : array-like
        Training data features
    y_train : array-like
        Training data target
    learning_rate : float, optional (default=0.1)
        Boosting learning rate
    n_estimators : int, optional (default=100)
        Number of boosted trees to fit
    max_depth : int, optional (default=6)
        Maximum tree depth for base learners
    num_leaves : int, optional (default=31)
        Maximum tree leaves for base learners
    subsample : float, optional (default=1.0)
        Subsample ratio of the training instance
    """
    model = lgb.LGBMRegressor(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        num_leaves=num_leaves,
        subsample=subsample,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_catboost(X_train, y_train, learning_rate=0.1, iterations=100, depth=6, l2_leaf_reg=3.0, border_count=128):
    """
    Train a CatBoost model
    
    Parameters:
    -----------
    X_train : array-like
        Training data features
    y_train : array-like
        Training data target
    learning_rate : float, optional (default=0.1)
        Boosting learning rate
    iterations : int, optional (default=100)
        Number of trees to build
    depth : int, optional (default=6)
        Depth of the trees
    l2_leaf_reg : float, optional (default=3.0)
        L2 regularization coefficient
    border_count : int, optional (default=128)
        Number of splits for numerical features
    """
    model = CatBoostRegressor(
        learning_rate=learning_rate,
        iterations=iterations,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        border_count=border_count,
        random_seed=42,
        verbose=False  # Disable training output
    )
    model.fit(X_train, y_train)
    return model 