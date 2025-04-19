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

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10):
    """
    Train a Random Forest model
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, learning_rate=0.1, n_estimators=100, max_depth=6):
    """
    Train an XGBoost model
    """
    model = xgb.XGBRegressor(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_lightgbm(X_train, y_train, learning_rate=0.1, n_estimators=100, max_depth=6):
    """
    Train a LightGBM model
    """
    model = lgb.LGBMRegressor(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_catboost(X_train, y_train, learning_rate=0.1, iterations=100, depth=6):
    """
    Train a CatBoost model
    """
    model = CatBoostRegressor(
        learning_rate=learning_rate,
        iterations=iterations,
        depth=depth,
        random_seed=42,
        verbose=False  # Disable training output
    )
    model.fit(X_train, y_train)
    return model 