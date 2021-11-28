import argparse
from pathlib import Path
from pprint import pprint
import os
import numpy as np
import pandas as pd

import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error

from preprocess import preprocess
from configs import *

import shap
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='Baseline',
                        help="config name in configs.py")
    opt = parser.parse_args()
    pprint(opt)

    ''' Configure path '''
    cfg = eval(opt.config)
    export_dir = Path('output') / cfg.name
    export_dir.mkdir(parents=True, exist_ok=True)

    ''' Prepare data '''
    train = pd.read_csv(cfg.train)
    test = pd.read_csv(cfg.test)
    train = train.replace("male",0).replace("female",1).replace("S",0).replace("C",1).replace("Q",2)
    test = test.replace("male",0).replace("female",1).replace("S",0).replace("C",1).replace("Q",2)
    
    train = preprocess(train, train)
    test = preprocess(test, train)

    train_data = train.values
    X = train_data[:, 2:]
    y  = train_data[:, 1]
    test_data = test.values
    X_test = test_data[:, 1:]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.2, random_state=0)
    lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train, free_raw_data=False)

    gbm = lgb.train(cfg.lgb_params,
                lgb_train,
                valid_sets=[lgb_train, lgb_eval],
                valid_names=['train', 'eval'],
                verbose_eval=10, 
                num_boost_round=1000, 
                early_stopping_rounds=10,
               )

    ''' Inference '''
    X_pred = gbm.predict(np.array(X_test), num_iteration=gbm.best_iteration)

    ''' SHAP Visualization'''
    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
    shap.initjs()
    explainer = shap.TreeExplainer(model=gbm, feature_perturbation='tree_path_dependent', model_output='margin')
    shap_values = explainer.shap_values(X=X_train)
    shap.summary_plot(shap_values, X_train, plot_type="bar")

if __name__ == '__main__':
    main()
