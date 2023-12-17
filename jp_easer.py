import os
from utils import *
import optuna
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender


data, usermap, itemmap, users = load_data2()
data_train, data_test, data_val = split_data2(data, 0, 0.2)


study_name = "Easer"
study = optuna.create_study(
    study_name=study_name,
    storage=get_database_url(),
    load_if_exists=True,
    direction="maximize",
)


def objective(trial):
    topK = trial.suggest_int('topK', 0, 1000)
    l2_norm = trial.suggest_float('l2_norm', 1, 1e10, log=True)
    
    recommender = EASE_R_Recommender(data_train, verbose=False)
    recommender.fit(
        topK=topK if topK > 0 else None,
        l2_norm=l2_norm,
        normalize_matrix=False,
    )
    _, _, ev_map, _, _ = evaluator(recommender, data_train, data_val)
    
    return ev_map


study.optimize(objective, n_trials=150)
