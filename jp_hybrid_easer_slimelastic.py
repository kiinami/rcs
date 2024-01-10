import os
from utils import *
import optuna
from Recommenders.BaseRecommender import BaseRecommender
import scipy.sparse as sps
import numpy.linalg as LA
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender

data, usermap, itemmap, users = load_data2()
data_train, data_val=split_data2(data, 0.2)

easer_study = optuna.create_study(
    study_name="Easer",
    storage=get_database_url(),
    load_if_exists=True,
    direction="maximize",
)

slimel_study = optuna.create_study(
    study_name="SLIMElastic",
    storage=get_database_url(),
    load_if_exists=True,
    direction="maximize",
)

easer_slimel_study = optuna.create_study(
    study_name="Easer+SLIMElastic",
    storage=get_database_url(),
    load_if_exists=True,
    direction="maximize",
)


class DifferentLossScoresHybridRecommender(BaseRecommender):
    """ ScoresHybridRecommender
    Hybrid of two prediction scores R = R1/norm*alpha + R2/norm*(1-alpha) where R1 and R2 come from
    algorithms trained on different loss functions.

    """

    RECOMMENDER_NAME = "DifferentLossScoresHybridRecommender"


    def __init__(self, URM_train, recommender_1, recommender_2):
        super(DifferentLossScoresHybridRecommender, self).__init__(URM_train)

        self.URM_train = sps.csr_matrix(URM_train)
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        
        
        
    def fit(self, norm, alpha = 0.5):

        self.alpha = alpha
        self.norm = norm


    def _compute_item_score(self, user_id_array, items_to_compute=None):
        
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)

        norm_item_weights_1 = LA.norm(item_weights_1, self.norm)
        norm_item_weights_2 = LA.norm(item_weights_2, self.norm)
        
        
        if norm_item_weights_1 == 0:
            norm_item_weights_1 = 0.0000001
        
        if norm_item_weights_2 == 0:
            norm_item_weights_2 = 0.0000001
        
        item_weights = item_weights_1 / norm_item_weights_1 * self.alpha + item_weights_2 / norm_item_weights_2 * (1-self.alpha)

        return item_weights



easer_recommender = EASE_R_Recommender(data_train, verbose=False)
easer_recommender.fit(**easer_study.best_params)

slimel_recommender = MultiThreadSLIM_SLIMElasticNetRecommender(data_train, verbose=False)
slimel_recommender.fit(**slimel_study.best_params)

def objective(trial):
    norm = trial.suggest_categorical("norm", [1, 2, np.inf, -np.inf, None])
    alpha = trial.suggest_float("alpha", 0, 1)

    recommender_object_2 = DifferentLossScoresHybridRecommender(data_train, easer_recommender, slimel_recommender)
    recommender_object_2.fit(
        norm = norm, 
        alpha = alpha
    )

    _, _, ev_map, _, _ = evaluator(recommender_object_2, data_train, data_val)
    
    return ev_map

easer_slimel_study.optimize(objective, n_trials=100)
