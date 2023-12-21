import os
from utils import *
import optuna
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender

data, usermap, itemmap, users = load_data2()
data_train, data_val = split_data2(data, 0.2)

study_name = "P3Beta"
study = optuna.create_study(
    study_name=study_name,
    storage=get_database_url(),
    load_if_exists=True,
    direction="maximize",
)

def objective(trial):
    topK = trial.suggest_int('topK', 1, 1000)
    alpha = trial.suggest_float('alpha', 0, 2)
    beta = trial.suggest_float('beta', 0, 2)
    implicit = trial.suggest_categorical('implicit', [True, False])
    normalize_similarity = trial.suggest_categorical('normalize_similarity', [True, False])
    
    recommender = RP3betaRecommender(data_train, verbose=False)
    recommender.fit(
        topK=topK,
        alpha=alpha,
        beta=beta,
        implicit=implicit,
        normalize_similarity=normalize_similarity,
    )
    _, _, ev_map, _, _ = evaluator(recommender, data_train, data_val)
    
    return ev_map

study.optimize(
    objective,
    n_trials=150,
)
