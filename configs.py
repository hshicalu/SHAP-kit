

class Baseline:
    
    name = 'baseline'
    seed = 2021
    train = "input/train.csv"
    test = "input/test.csv"

    lgb_params = {
        'boosting_type': 'gbdt',
        'max_depth': -1,
        'objective': 'binary',
        'num_leaves': 64,
        'learning_rate': 0.05,
        'max_bin': 512,
        'subsample_for_bin': 200,
        'subsample': 1,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
        'reg_alpha': 5,
        'reg_lambda': 10,
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 5,
        'scale_pos_weight': 1,
        'num_class': 1,
        'metric': 'binary_error',
         }
    amp = True
    parallel = None
    deterministic = False