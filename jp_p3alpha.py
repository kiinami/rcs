import os
from utils import *
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender

data, usermap, itemmap, users = load_data2()
data_train, data_val = split_data2(data, 0.2)

study_name = "P3Alpha"
study = optuna.create_study(
    study_name=study_name,
    storage=get_database_url(),
    load_if_exists=True,
    direction="maximize",
)

def objective(trial):
    topK = trial.suggest_int('topK', 1, 1000)
    alpha = trial.suggest_float('alpha', 0, 2)
    implicit = trial.suggest_categorical('implicit', [True, False])
    normalize_similarity = trial.suggest_categorical('normalize_similarity', [True, False])
    
    recommender = P3alphaRecommender(data_train, verbose=False)
    recommender.fit(
        topK=topK,
        alpha=alpha,
        implicit=implicit,
        normalize_similarity=normalize_similarity,
    )
    _, _, ev_map, _, _ = evaluator(recommender, data_train, data_val)
    
    return ev_map

study.optimize(
    objective,
    n_trials=150,
)
