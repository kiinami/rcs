import os
from utils import *
import optuna
from Recommenders.MatrixFactorization.NMFRecommender import NMFRecommender


data, usermap, itemmap, users = load_data2()
data_train, data_val = split_data2(data, 0.2)


study_name = "NFM"
study = optuna.create_study(
    study_name=study_name,
    storage=get_database_url(),
    load_if_exists=True,
    direction="maximize",
)

def objective(trial):
    num_factors = trial.suggest_int('num_factors', 1, 1000)
    l1_ratio = trial.suggest_float('l1_ratio', 0, 1)
    solver = trial.suggest_categorical('solver', ['coordinate_descent', 'multiplicative_update'])
    init_type = trial.suggest_categorical('init_type', ['random', 'nndsvda'])
    beta_loss = trial.suggest_categorical('beta_loss', ['frobenius', 'kullback-leibler'])
    
    recommender = NMFRecommender(data_train, verbose=False)
    recommender.fit(
        num_factors=num_factors,
        l1_ratio=l1_ratio,
        solver=solver,
        init_type=init_type,
        beta_loss=beta_loss,
        verbose=False
    )
    _, _, ev_map, _, _ = evaluator(recommender, data_train, data_val)
    
    return ev_map

study.optimize(objective, n_trials=150)
