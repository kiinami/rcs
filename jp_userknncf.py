import os
from utils import *
import optuna
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender

data, usermap, itemmap, users = load_data2()
data_train, data_val = split_data2(data, 0.2)

study_name = "UserKNNCF"
study = optuna.create_study(
    study_name=study_name,
    storage=get_database_url(),
    load_if_exists=True,
    direction="maximize",
)


def objective(trial):
    topK = trial.suggest_int('topK', 0, 1000)
    shrink = trial.suggest_int('topK', 0, 1000)
    similarity = trial.suggest_categorical('similarity', ['cosine', 'jaccard', 'asymmetric', 'dice', 'tversky'])
    normalize = trial.suggest_categorical('normalize', [True, False])
    feature_weighting = trial.suggest_categorical('feature_weighting', ['BM25', 'TF-IDF', 'none'])
    
    recommender = UserKNNCFRecommender(data_train, verbose=False)
    recommender.fit(
        topK=topK,
        shrink=shrink,
        similarity=similarity,
        normalize=normalize,
        feature_weighting=feature_weighting,
    )
    _, _, ev_map, _, _ = evaluator(recommender, data_train, data_val)
    
    return ev_map

study.optimize(objective, n_trials=300)
