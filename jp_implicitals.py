from utils import *
import optuna
from implicit.gpu.als import AlternatingLeastSquares as IALSRecommender
from implicit.evaluation import ranking_metrics_at_k

data, usermap, itemmap, users = load_data2()
data_train, data_test, data_val = split_data2(data, 0, 0.2)

study_name = "ImplicitALS"
study = optuna.create_study(
    study_name=study_name,
    storage=get_database_url(),
    load_if_exists=True,
    direction="maximize",
)

def objective(trial):
    num_factors = trial.suggest_int('num_factors', 1, 500)
    reg = trial.suggest_float('reg', 1e-5, 1, log=True)
    alpha = trial.suggest_float('alpha', 1e-7, 100, log=True)
    iterations = trial.suggest_int('iterations', 10, 1000)
    
    recommender = IALSRecommender(
        factors=num_factors,
        regularization=reg,
        alpha=alpha,
        iterations=iterations,
        calculate_training_loss=False,
    )
    recommender.fit(
        data_train,
        show_progress=False,
    )
    
    return ranking_metrics_at_k(recommender, data_train, data_val, K=10, show_progress=False)['map']

study.optimize(objective, n_trials=300)
