# Q2/prediction_intervals.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import cross_val_score
import lightgbm as lgb

class PredictionIntervalModel:
    def __init__(self, alpha=0.1):
        """
        alpha: significance level (0.1 for 90% confidence interval)
        """
        self.alpha = alpha
        self.lower_quantile = alpha/2
        self.upper_quantile = 1 - alpha/2
        
    def quantile_regression_approach(self, X_train, y_train, X_test):
        """Phương pháp Quantile Regression"""
        # Model cho quantile dưới
        lower_model = QuantileRegressor(quantile=self.lower_quantile, alpha=0.01)
        lower_model.fit(X_train, y_train)
        
        # Model cho quantile trên  
        upper_model = QuantileRegressor(quantile=self.upper_quantile, alpha=0.01)
        upper_model.fit(X_train, y_train)
        
        # Dự đoán
        lower_pred = lower_model.predict(X_test)
        upper_pred = upper_model.predict(X_test)
        
        return lower_pred, upper_pred
    
    def lightgbm_quantile_approach(self, X_train, y_train, X_test):
        """Phương pháp LightGBM với quantile objective"""
        # Model cho quantile dưới
        lower_params = {
            'objective': 'quantile',
            'alpha': self.lower_quantile,
            'metric': 'quantile',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        lower_model = lgb.train(lower_params, train_data, num_boost_round=1000)
        
        # Model cho quantile trên
        upper_params = lower_params.copy()
        upper_params['alpha'] = self.upper_quantile
        
        upper_model = lgb.train(upper_params, train_data, num_boost_round=1000)
        
        # Dự đoán
        lower_pred = lower_model.predict(X_test)
        upper_pred = upper_model.predict(X_test)
        
        return lower_pred, upper_pred
    
    def ensemble_approach(self, X_train, y_train, X_test):
        """Phương pháp Ensemble với nhiều models"""
        models = []
        predictions = []
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        
        # Gradient Boosting
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb.fit(X_train, y_train)
        gb_pred = gb.predict(X_test)
        
        # Tính prediction intervals từ ensemble
        ensemble_pred = (rf_pred + gb_pred) / 2
        
        # Ước tính uncertainty
        pred_std = np.std([rf_pred, gb_pred], axis=0)
        z_score = 1.645  # for 90% confidence interval
        
        lower_pred = ensemble_pred - z_score * pred_std
        upper_pred = ensemble_pred + z_score * pred_std
        
        return lower_pred, upper_pred
