import os
from utils import *
import optuna
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_SVDpp_Cython

data, usermap, itemmap, users = load_data2()
data_train, data_val = split_data2(data, 0.2)

study_name = "SVDpp"
study = optuna.create_study(
    study_name=study_name,
    storage=get_database_url(),
    load_if_exists=True,
    direction="maximize",
)


def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-8, 1)
    user_reg = trial.suggest_float('user_reg', 1e-8, 1)
    item_reg = trial.suggest_float('item_reg', 1e-8, 1)
    epochs = trial.suggest_int('epochs', 5, 500)
    
    recommender = MatrixFactorization_SVDpp_Cython(data_train, verbose=False)
    recommender.fit(
        learning_rate=learning_rate,
        user_reg=user_reg,
        item_reg=item_reg,
        epochs=epochs
    )
    _, _, ev_map, _, _ = evaluator(recommender, data_train, data_val)
    
    return ev_map

study.optimize(objective, n_trials=500)
