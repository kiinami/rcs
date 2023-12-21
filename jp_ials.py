import os
from utils import *
import optuna
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender


data, usermap, itemmap, users = load_data2()
data_train, data_val = split_data2(data, 0.2)


study_name = "IALS"
study = optuna.create_study(
    study_name=study_name,
    storage=get_database_url(),
    load_if_exists=True,
    direction="maximize",
)

evaluator_validation = EvaluatorHoldout(data_val, cutoff_list=[10])

def objective(trial):
    num_factors = trial.suggest_int('num_factors', 1, 500)
    epochs = trial.suggest_categorical('epochs', [50, 100, 200, 300, 400])
    confidence_scaling = trial.suggest_categorical('confidence_scaling', ["linear", "log"])
    alpha = trial.suggest_float('alpha', 1e-7, 100, log=True)
    epsilon = trial.suggest_float('epsilon', 1e-7, 100, log=True)
    reg = trial.suggest_float('reg', 1e-5, 1, log=True)
    
    recommender = IALSRecommender(data_train, verbose=False)
    recommender.fit(
        epochs=epochs,
        num_factors=num_factors,
        confidence_scaling=confidence_scaling,
        alpha=alpha,
        epsilon=epsilon,
        reg=reg,
        validation_every_n=5,
        stop_on_validation=True,
        evaluator_object=evaluator_validation,
        lower_validations_allowed=5,
        validation_metric="MAP",
    )
    _, _, ev_map, _, _ = evaluator(recommender, data_train, data_val)
    
    return ev_map


study.optimize(objective, n_trials=150)
