import optuna
from utils import *
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython

data, usermap, itemmap, users = load_data2()
data_train, data_val = split_data2(data, 0.2)

study = optuna.create_study(
    study_name="SLIMBPR",
    storage=get_database_url(),
    load_if_exists=True,
    direction="maximize",
)

def objective(trial):
    epochs = trial.suggest_int('epochs', 1, 500, log=True)
    symmetric = trial.suggest_categorical('symmetric', [True, False])
    lambda_i = trial.suggest_float('lambda_i', 1e-8, 1, log=True)
    lambda_j = trial.suggest_float('lambda_j', 1e-8, 1, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-8, 1, log=True)
    topK = trial.suggest_int('topK', 1, 1000, log=True)
    sgd_mode = trial.suggest_categorical('sgd_mode', ["sgd", "adagrad", "adam"])
    gamma = trial.suggest_float('gamma', 1e-8, 1, log=True)
    beta_1 = trial.suggest_float('beta_1', 1e-8, 1, log=True)
    beta_2 = trial.suggest_float('beta_2', 1e-8, 1, log=True)
    
    recommender = SLIM_BPR_Cython(data_train, verbose=False)
    recommender.fit(
        epochs=epochs,
        symmetric=symmetric,
        lambda_i=lambda_i,
        lambda_j=lambda_j,
        learning_rate=learning_rate,
        topK=topK,
        sgd_mode=sgd_mode,
        gamma=gamma,
        beta_1=beta_1,
        beta_2=beta_2,
    )
    _, _, ev_map, _, _ = evaluator(recommender, data_train, data_val)
    
    return ev_map

study.optimize(objective, n_trials=1000)
