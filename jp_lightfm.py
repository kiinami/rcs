#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from utils import *
import optuna


# In[2]:


data, usermap, itemmap, users = load_data2()
data_train, data_test, data_val = split_data2(data, 0, 0.2)


# In[ ]:


study_name = "WARP"
study = optuna.create_study(
    study_name=study_name,
    storage=get_database_url(),
    load_if_exists=True,
    direction="maximize",
)


# In[3]:


from Recommenders.BaseRecommender import BaseRecommender
from lightfm import LightFM
import numpy as np

class LightFMCFRecommender(BaseRecommender):
    """LightFMCFRecommender"""

    RECOMMENDER_NAME = "LightFMCFRecommender"

    def __init__(self, URM_train):
        super(LightFMCFRecommender, self).__init__(URM_train)


    def fit(self, epochs = 300, alpha = 1e-6, n_factors = 10, n_threads = 4):
        
        # Let's fit a WARP model
        self.lightFM_model = LightFM(loss='warp',
                                     item_alpha=alpha,
                                     no_components=n_factors)

        self.lightFM_model = self.lightFM_model.fit(self.URM_train, 
                                       epochs=epochs,
                                       num_threads=n_threads)


    def _compute_item_score(self, user_id_array, items_to_compute = None):
        
        # Create a single (n_items, ) array with the item score, then copy it for every user
        items_to_compute = np.arange(self.n_items)
        
        item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf

        for user_index, user_id in enumerate(user_id_array):
            item_scores[user_index] = self.lightFM_model.predict(int(user_id), 
                                                                 items_to_compute)

        return item_scores


# In[ ]:


def objective(trial):
    epochs = trial.suggest_int('epochs', 1, 1000)
    alpha = trial.suggest_float('alpha', 0, 1)
    n_factors = trial.suggest_int('n_factors', 1, 100)
    
    recommender = LightFMCFRecommender(data_train)
    recommender.fit(
        epochs=epochs, 
        alpha=alpha, 
        n_factors=n_factors, 
        n_threads=12
    )
    _, _, ev_map, _, _ = evaluator(recommender, data_train, data_val)
    
    return ev_map

study.optimize(objective, n_trials=150)


# In[ ]:




