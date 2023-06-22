from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression, OrthogonalMatchingPursuit, BayesianRidge

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import numpy as np

scoring = 'neg_mean_squared_error'
cv = KFold(n_splits=5)

def gbr(trial, X, y, scoring=scoring, cv=cv):
    n_estimators = trial.suggest_int('n_estimators', 50, 1000)
    learning_rate = trial.suggest_uniform('learning_rate', 0.001, 0.1)
    max_depth = trial.suggest_int('max_depth', 2, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    
    model = GradientBoostingRegressor(n_estimators=n_estimators,
                                       learning_rate=learning_rate,
                                       max_depth=max_depth,
                                       min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf)
    
    model.fit(X, y)
    
    if scoring=='neg_mean_squared_error':
        np.exp(np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)))
    elif scoring=='neg_mean_absolute_error':
        cv_scores = (-cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv))
    else:
        cv_scores = cross_val_score(model, X, y, scoring=X, y, scoring=scoring, cv=cv=cv)
        
    return np.mean(cv_scores)

def lgbm(trial, X, y, scoring=scoring, cv=cv):
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 1),
        'subsample': trial.suggest_float('subsample', 0.1, 1),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 1),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'max_depth': trial.suggest_int('max_depth', 2, 30),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'min_gain_to_split': trial.suggest_loguniform('min_gain_to_split', 0.1, 5),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'random_state': 42
    }
    
    model = LGBMRegressor(**params)
    
    model.fit(X, y)
    
    if scoring=='neg_mean_squared_error':
        np.exp(np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)))
    elif scoring=='neg_mean_absolute_error':
        cv_scores = (-cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv))
    else:
        cv_scores = cross_val_score(model, X, y, scoring=X, y, scoring=scoring, cv=cv=cv)
        
    return np.mean(cv_scores)

def xgb(trial, X, y, scoring=scoring, cv=cv):
    n_estimators = trial.suggest_int('n_estimators', 50, 1000)
    learning_rate = trial.suggest_uniform('learning_rate', 0.001, 0.1)
    max_depth = trial.suggest_int('max_depth', 2, 10)
    subsample = trial.suggest_uniform('subsample', 0.1, 1.0)
    colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.1, 1.0)
    
    model = XGBRegressor(n_estimators=n_estimators,
                             learning_rate=learning_rate,
                             max_depth=max_depth,
                             subsample=subsample,
                             colsample_bytree=colsample_bytree)
    
    model.fit(X, y)
    
    if scoring=='neg_mean_squared_error':
        np.exp(np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)))
    elif scoring=='neg_mean_absolute_error':
        cv_scores = (-cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv))
    else:
        cv_scores = cross_val_score(model, X, y, scoring=X, y, scoring=scoring, cv=cv=cv)
        
    return np.mean(cv_scores)

def rfr(trial, X, y, scoring=scoring, cv=cv):
    n_estimators = trial.suggest_int('n_estimators', 50, 1000, step=100)
    max_depth = trial.suggest_int('max_depth', 2, 30, step=2)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_uniform('max_features', 0.1, 1.0)
    
    model = RandomForestRegressor(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   max_features=max_features)

    if scoring=='neg_mean_squared_error':
        np.exp(np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)))
    elif scoring=='neg_mean_absolute_error':
        cv_scores = (-cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv))
    else:
        cv_scores = cross_val_score(model, X, y, scoring=X, y, scoring=scoring, cv=cv=cv)
        
    return np.mean(cv_scores)

def lr(trial, X, y, scoring=scoring, cv=cv):
    fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
    
    model = LinearRegression(fit_intercept=fit_intercept)
    
    model.fit(X, y)
    
    if scoring=='neg_mean_squared_error':
        np.exp(np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)))
    elif scoring=='neg_mean_absolute_error':
        cv_scores = (-cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv))
    else:
        cv_scores = cross_val_score(model, X, y, scoring=X, y, scoring=scoring, cv=cv=cv)
        
    return np.mean(cv_scores)

def ridge(trial, X, y, scoring=scoring, cv=cv):
    alpha = trial.suggest_int('alpha', 0, 1000)
    tol = trial.suggest_loguniform('tol', 1e-8, 10.0)
        
    model = Ridge(
        alpha=alpha,
        tol=tol
    )
    
    model.fit(train_final, log_target)
    
    if scoring=='neg_mean_squared_error':
        np.exp(np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)))
    elif scoring=='neg_mean_absolute_error':
        cv_scores = (-cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv))
    else:
        cv_scores = cross_val_score(model, X, y, scoring=X, y, scoring=scoring, cv=cv=cv)
        
    return np.mean(cv_scores)

def br(trial, X, y, scoring=scoring, cv=cv):
    n_iter = trial.suggest_int('n_iter', 50, 600)
    tol = trial.suggest_loguniform('tol', 1e-8, 10.0)
    alpha_1 = trial.suggest_loguniform('alpha_1', 1e-8, 10.0)
    alpha_2 = trial.suggest_loguniform('alpha_2', 1e-8, 10.0)
    lambda_1 = trial.suggest_loguniform('lambda_1', 1e-8, 10.0)
    lambda_2 = trial.suggest_loguniform('lambda_2', 1e-8, 10.0)
    
    model = BayesianRidge(
        n_iter=n_iter,
        tol=tol,
        alpha_1=alpha_1,
        alpha_2=alpha_2,
        lambda_1=lambda_1,
        lambda_2=lambda_2
    )
    
    model.fit(train_final, log_target)
    
    if scoring=='neg_mean_squared_error':
        np.exp(np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)))
    elif scoring=='neg_mean_absolute_error':
        cv_scores = (-cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv))
    else:
        cv_scores = cross_val_score(model, X, y, scoring=X, y, scoring=scoring, cv=cv=cv)
        
    return np.mean(cv_scores)

def omp(trial, X, y, scoring=scoring, cv=cv):
    tol = trial.suggest_loguniform('tol', 1e-8, 10.0)
    
    model = OrthogonalMatchingPursuit(
        tol=tol
    )
    
    model.fit(train_final, log_target)
    
    if scoring=='neg_mean_squared_error':
        np.exp(np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)))
    elif scoring=='neg_mean_absolute_error':
        cv_scores = (-cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv))
    else:
        cv_scores = cross_val_score(model, X, y, scoring=X, y, scoring=scoring, cv=cv=cv)
        
    return np.mean(cv_scores)
