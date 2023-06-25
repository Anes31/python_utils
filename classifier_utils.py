from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import numpy as np

scoring = 'accuracy'
cv = KFold(n_splits=5, shuffle=True, random_state=0)

def lightgbm_multi(trial, X, y, scoring=scoring, cv=cv):
    params = {
        'objective': 'multiclass',
        'boosting_type': 'gbdt',
        'metric': 'multi_logloss',
        'n_estimators': trial.suggest_int('n_estimators', 10, 500),
        'max_depth': trial.suggest_int('max_depth', 1, 50),
        'num_leaves': trial.suggest_int('num_leaves', 2, 100),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'subsample': trial.suggest_uniform('subsample', 0.1, 1),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-9, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-9, 10.0),
        'verbose': -1,
        'random_state': 0
    }

    model = LGBMClassifier(**params)

    model.fit(X, y)
    
    cv_scores = cross_val_score(model, X, y, scoring=scoring, cv=cv)
    
    return np.mean(cv_scores)

def lightgbm_binary(trial, X, y, scoring=scoring, cv=cv):
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'binary_logloss',
        'n_estimators': trial.suggest_int('n_estimators', 10, 500),
        'max_depth': trial.suggest_int('max_depth', 1, 50),
        'num_leaves': trial.suggest_int('num_leaves', 2, 100),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'subsample': trial.suggest_uniform('subsample', 0.1, 1),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-9, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-9, 10.0),
        'verbose': -1,
        'random_state': 0
    }

    model = LGBMClassifier(**params)

    model.fit(X, y)
    
    cv_scores = cross_val_score(model, X, y, scoring=scoring, cv=cv)
    
    return np.mean(cv_scores)

def xgb(trial, X, y, scoring=scoring, cv=cv):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'subsample': trial.suggest_uniform('subsample', 0.1, 1),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-10, 1),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-10, 1),
        'gamma': trial.suggest_loguniform('gamma', 1e-10, 1),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'objective': 'binary:logistic', #'mlogloss', 
        'eval_metric': 'logloss',
        'verbosity': 0,
        'random_state': 0,
        'n_jobs': -1
    }
    
    model = XGBClassifier(**params)

    model.fit(X, y)
    
    cv_scores = cross_val_score(model, X, y, scoring=scoring, cv=cv)
    
    return np.mean(cv_scores)

def catboost(trial, X, y, scoring=scoring, cv=cv):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 1, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0),
        'random_seed': 0,
        #'loss_function': 'MultiClass',
        #'eval_metric': 'MultiClass',
        'logging_level': 'Silent'
    }
    
    model = CatBoostClassifier(**params)
    
    model.fit(X, y)
    
    cv_scores = cross_val_score(model, X, y, scoring=scoring, cv=cv)
    
    return np.mean(cv_scores)

def rf(trial, X, y, scoring=scoring, cv=cv):
    max_depth = trial.suggest_int('max_depth', 1, 100)
    n_estimators = trial.suggest_int('n_estimators', 10, 500)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 100)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 100)
    random_state = 0
          
    model = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=random_state
    )
    
    model.fit(X, y)
    
    cv_scores = cross_val_score(model, X, y, scoring=scoring, cv=cv)
    
    return np.mean(cv_scores)

def gbc(trial, X, y, scoring=scoring, cv=cv):
    #tol = trial.suggest_loguniform('tol', 1e-8, 10.0)
    max_depth = trial.suggest_int('max_depth', 1, 50)
    learning_rate = trial.suggest_loguniform('learning_rate', .001, 1)
    n_estimators = trial.suggest_int('n_estimators', 10, 500)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 100)
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 2, 100)
    random_state = 0
          
    model = GradientBoostingClassifier(
        #tol=tol,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_leaf_nodes=max_leaf_nodes,
        random_state=random_state
    )
    
    model.fit(X, y)
    
    cv_scores = cross_val_score(model, X, y, scoring=scoring, cv=cv)
    
    return np.mean(cv_scores)

def et(trial, X, y, scoring=scoring, cv=cv):
    max_depth = trial.suggest_int('max_depth', 1, 100)
    n_estimators = trial.suggest_int('n_estimators', 10, 500)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 100)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 100)
    max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None])
    random_state = 0

          
    model = ExtraTreesClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_features=max_features,
        random_state=random_state
    )
    
    model.fit(X, y)
    
    cv_scores = cross_val_score(model, X, y, scoring=scoring, cv=cv)
    
    return np.mean(cv_scores)

def lr(trial, X, y, scoring=scoring, cv=cv):
    C = trial.suggest_loguniform('C', 0.001, 1000)
    solver = trial.suggest_categorical('solver', ['newton-cg', 'liblinear', 'sag', 'saga'])
    
    if solver=='sag' or solver=='newton-cg':
        penalty = 'l2'
        multi_class = trial.suggest_categorical('multi_class', ['ovr', 'multinomial'])
    elif solver=='liblinear':
        multi_class = 'ovr'
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
        if penalty==1:
            l1_ratio = trial.suggest_float(0, 1)
    elif solver=='saga':
        multi_class = trial.suggest_categorical('multi_class', ['ovr', 'multinomial'])
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
        if penalty==1:
            l1_ratio = trial.suggest_float(0, 1)
          
    model = LogisticRegression(
        C=C,
        penalty = penalty,
        solver=solver,
        multi_class=multi_class
    )
    
    model.fit(X, y)
    
    cv_scores = cross_val_score(model, X, y, scoring=scoring, cv=cv)
    
    return np.mean(cv_scores)
    
def ridge(trial, X, y, scoring=scoring, cv=cv):
    alpha = trial.suggest_int('alpha', 0, 1000)
    tol = trial.suggest_loguniform('tol', 1e-8, 10.0)
        
    model = RidgeClassifier(
        alpha=alpha,
        tol=tol
    )
    calibrated_model = CalibratedClassifierCV(model, method='sigmoid')
    
    calibrated_model.fit(X, y)
    
    cv_scores = cross_val_score(model, X, y, scoring=scoring, cv=cv)
    
    return np.mean(cv_scores)

def lda(trial, X, y, scoring=scoring, cv=cv):
    solver = trial.suggest_categorical('solver', ['lsqr', 'eigen'])
    tol = trial.suggest_loguniform('tol', 1e-8, 10.0)
          
    model = LinearDiscriminantAnalysis(
        solver=solver,
        tol=tol
    )
    
    model.fit(X, y)
    
    cv_scores = cross_val_score(model, X, y, scoring=scoring, cv=cv)
    
    return np.mean(cv_scores)

def nb(trial, X, y, scoring=scoring, cv=cv):
    var_smoothing  = trial.suggest_loguniform('var_smoothing', 1e-10, 1e-3)
          
    model = GaussianNB(
        var_smoothing=var_smoothing
    )
    
    model.fit(X, y)
    
    cv_scores = cross_val_score(model, X, y, scoring=scoring, cv=cv)
    
    return np.mean(cv_scores)

def ada(trial, X, y, scoring=scoring, cv=cv):
    learning_rate = trial.suggest_loguniform('learning_rate', .001, 1)
    n_estimators = trial.suggest_int('n_estimators', 10, 500)
          
    model = AdaBoostClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators
    )
    
    model.fit(train_final, num_target)
    
    cv_scores = cross_val_score(model, X, y, scoring=scoring, cv=cv)
    
    return np.mean(cv_scores)

def knn(trial, X, y, scoring=scoring, cv=cv):
    n_neighbors = trial.suggest_int('n_neighbors', 2, 100)
          
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors
    )
    
    model.fit(X, y)
    
    cv_scores = cross_val_score(model, X, y, scoring=scoring, cv=cv)
    
    return np.mean(cv_scores)