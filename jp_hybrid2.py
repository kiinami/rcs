#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from utils import *
import optuna
from Recommenders.Recommender_import_list import *
from Recommenders.BaseRecommender import BaseRecommender
import scipy.sparse as sps


# In[2]:


data, usermap, itemmap, users = load_data2()
data_train, data_val = split_data2(data, 0.2)


# In[3]:


# 0.08254950935282762
slimel_study = optuna.load_study(
    study_name="SLIMElastic",
    storage=get_database_url(),
)

# 0.07018143152516958
slimbpr_study = optuna.load_study(
    study_name="SLIMBPR",
    storage=get_database_url(),
)

# 0.08248515814658722
p3alpha_study = optuna.load_study(
    study_name="P3Alpha",
    storage=get_database_url(),
)

# 0.082429130748692
easer_study = optuna.load_study(
    study_name="Easer",
    storage=get_database_url(),
)

# 0.06470798788105654
userknncf_study = optuna.load_study(
    study_name="UserKNNCF",
    storage=get_database_url(),
)

# 0.07684279728158312
itemknncf_study = optuna.load_study(
    study_name="ItemKNNCF",
    storage=get_database_url(),
)

# 0.08511222934213017
p3beta_study = optuna.load_study(
    study_name="P3Beta",
    storage=get_database_url(),
)


# In[4]:


recommenders = [
    {
        "recommender": ItemKNNCFRecommender,
        "params": itemknncf_study.best_params,
    },
    {
        "recommender": UserKNNCFRecommender,
        "params": userknncf_study.best_params,
    },
    {
        "recommender": RP3betaRecommender,
        "params": p3beta_study.best_params,
    },
    {
        "recommender": EASE_R_Recommender,
        "params": easer_study.best_params,
    },
    {
        "recommender": P3alphaRecommender,
        "params": p3alpha_study.best_params,
    },
    {
        "recommender": SLIM_BPR_Cython,
        "params": slimbpr_study.best_params,
    },
    {
        "recommender": SLIMElasticNetRecommender,
        "params": slimel_study.best_params,
    },
]


# In[5]:


from Recommenders.BaseRecommender import BaseRecommender

class ScoresHybridRecommender(BaseRecommender):

    RECOMMENDER_NAME = "ScoresHybridRecommender"

    def __init__(self, data_train, recommenders):
        super(ScoresHybridRecommender, self).__init__(data_train)

        self.data_train = sps.csr_matrix(data_train)
        self.recommenders = recommenders
        
    def prefit(self):
        for rec in self.recommenders:
            rec["recommender"] = rec["recommender"](self.data_train)
            rec["recommender"].fit(**rec["params"])
        
        
    def fit(self, weights):
        self.weights = weights      


    def _compute_item_score(self, user_id_array, items_to_compute):
        
        # In a simple extension this could be a loop over a list of pretrained recommender objects
        item_weights = []
        for rec in self.recommenders:
            item_weights.append(rec["recommender"]._compute_item_score(user_id_array, items_to_compute))

        item_weights = sum([a*b for a,b in zip(item_weights, self.weights)])
        return item_weights


# In[6]:


scores_hybrid_study = optuna.create_study(
    study_name="ScoresHybrid2",
    storage=get_database_url(),
    load_if_exists=True,
    direction="maximize",
)


# In[7]:


scores_hybrid_recommender = ScoresHybridRecommender(data_train, recommenders)
scores_hybrid_recommender.prefit()

def objective(trial):
    weights = []
    for i in range(len(recommenders)):
        weights.append(trial.suggest_uniform("weight_{}".format(i), 0, 1))
    scores_hybrid_recommender.fit(weights)
    _, _, ev_map, _, _ = evaluator(scores_hybrid_recommender, data_train, data_val)
    return ev_map

scores_hybrid_study.optimize(objective, n_trials=200)


# In[ ]:




