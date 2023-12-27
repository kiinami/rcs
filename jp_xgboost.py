import os
from utils import *
import optuna
from Recommenders.Recommender_import_list import *
import scipy.sparse as sps
from xgboost import XGBRanker

# Load all data
data, usermap, itemmap, users = load_data2()
data_train, data_val = split_data2(data, 0.2)

# Creates the dataframe with the initial recommendations
cutoff = 50

candidate_recommender_study = optuna.create_study(
    study_name="P3Beta",
    storage=get_database_url(),
    load_if_exists=True,
    direction="maximize",
)
candidate_recommender = RP3betaRecommender(data_train, verbose=False)
candidate_recommender.fit(**candidate_recommender_study.best_params)
n_users, n_items = data_train.shape
training_dataframe = pd.DataFrame(index=range(0,n_users), columns = ["ItemID"])
training_dataframe.index.name='UserID'
for user_id in range(n_users):    
    recommendations = candidate_recommender.recommend(user_id, cutoff = cutoff)
    training_dataframe.loc[user_id, "ItemID"] = recommendations
training_dataframe = training_dataframe.explode("ItemID")
data_val_coo = sps.coo_matrix(data_val)
correct_recommendations = pd.DataFrame({"UserID": data_val_coo.row,
                                        "ItemID": data_val_coo.col})
training_dataframe = pd.merge(training_dataframe, correct_recommendations, on=['UserID','ItemID'], how='left', indicator='Exist')
training_dataframe["Label"] = training_dataframe["Exist"] == "both"
training_dataframe.drop(columns = ['Exist'], inplace=True)

# Adds more data to the dataframe
topPop = TopPop(data_train)
topPop.fit()

p3alpha_study = optuna.create_study(
    study_name="P3Alpha",
    storage=get_database_url(),
    load_if_exists=True,
    direction="maximize",
)
p3alpha = P3alphaRecommender(data_train, verbose=False)
p3alpha.fit(**p3alpha_study.best_params)

easer_study = optuna.create_study(
    study_name="Easer",
    storage=get_database_url(),
    load_if_exists=True,
    direction="maximize",
)
easer = EASE_R_Recommender(data_train, verbose=False)
easer.fit(**easer_study.best_params)

itemknncf_study = optuna.create_study(
    study_name="ItemKNNCF",
    storage=get_database_url(),
    load_if_exists=True,
    direction="maximize",
)
itemknncf = ItemKNNCFRecommender(data_train, verbose=False)
itemknncf.fit(**itemknncf_study.best_params)

other_algorithms = {
    "TopPop": topPop,
    "P3alpha": p3alpha,
    "Easer": easer,
    "ItemKNNCF": itemknncf
}

training_dataframe = training_dataframe.set_index('UserID')
for user_id in range(n_users):  
    for rec_label, rec_instance in other_algorithms.items():
        item_list = training_dataframe.loc[user_id, "ItemID"].values.tolist()
        all_item_scores = rec_instance._compute_item_score([user_id], items_to_compute = item_list)
        training_dataframe.loc[user_id, rec_label] = all_item_scores[0, item_list] 
training_dataframe = training_dataframe.reset_index()
training_dataframe = training_dataframe.rename(columns = {"index": "UserID"})
item_popularity = np.ediff1d(sps.csc_matrix(data_train).indptr)
training_dataframe['item_popularity'] = item_popularity[training_dataframe["ItemID"].values.astype(int)]
user_popularity = np.ediff1d(sps.csr_matrix(data_train).indptr)
training_dataframe['user_profile_len'] = user_popularity[training_dataframe["UserID"].values.astype(int)]

training_dataframe = training_dataframe.set_index('ItemID')
training_dataframe = training_dataframe.reset_index()
training_dataframe = training_dataframe.rename(columns = {"index": "ItemID"})

groups = training_dataframe.groupby("UserID").size().values
train_recs = training_dataframe.drop(columns=["Label"])
train_labs = training_dataframe["Label"]

# Starts the training
study = optuna.create_study(
    study_name="XGBoost(P3Beta, TopPop, P3alpha, Easer, ItemKNNCF)",
    storage=get_database_url(),
    load_if_exists=True,
    direction="maximize",
)

def evaluator(data_train, train_recs, XGB_model, training_dataframe, data_test: sp.csr_matrix, recommendation_length: int = 10):
    accum_precision = 0
    accum_recall = 0
    accum_map = 0

    num_users = data_train.shape[0]

    num_users_evaluated = 0
    num_users_skipped = 0
    for user_id in range(num_users):
        user_profile_start = data_test.indptr[user_id]
        user_profile_end = data_test.indptr[user_id+1]

        relevant_items = data_test.indices[user_profile_start:user_profile_end]

        if relevant_items.size == 0:
            num_users_skipped += 1
            continue
        
        predict = train_recs[train_recs["UserID"] == user_id]
        predictions = XGB_model.predict(predict)
        user_ids = predict["UserID"].values
        item_ids = predict["ItemID"].values
        predictions = pd.DataFrame({"UserID": user_ids, "ItemID": item_ids, "Predictions": predictions})
        predictions = predictions.sort_values(by="Predictions", ascending=False)
        recommendations = np.array(predictions["ItemID"].values[:recommendation_length])

        accum_precision += precision(recommendations, relevant_items)
        accum_recall += recall(recommendations, relevant_items)
        accum_map += mean_average_precision(recommendations, relevant_items)

        num_users_evaluated += 1


    accum_precision /= max(num_users_evaluated, 1)
    accum_recall /= max(num_users_evaluated, 1)
    accum_map /=  max(num_users_evaluated, 1)

    return accum_precision, accum_recall, accum_map, num_users_evaluated, num_users_skipped


def objective(trial):
    objective = trial.suggest_categorical("objective", ["rank:ndcg", "rank:map", "rank:pairwise"])
    n_estimators = trial.suggest_int("n_estimators", 5, 500)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    reg_alpha = trial.suggest_float("reg_alpha", 1e-4, 1e-1, log=True)
    reg_lambda = trial.suggest_float("reg_lambda", 1e-4, 1e-1, log=True)
    max_depth = trial.suggest_int("max_depth", 1, 10)
    max_leaves = trial.suggest_int("max_leaves", 1, 10)
    grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    booster = trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"])

    XGB_model = XGBRanker(
        tree_method="gpu_hist",
        predictor='gpu_predictor',
        objective=objective,
        n_estimators=int(n_estimators),
        random_state=None,
        learning_rate=learning_rate,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        max_depth=int(max_depth),
        max_leaves=int(max_leaves),
        grow_policy=grow_policy,
        booster=booster,
        verbosity=0, # 2 if self.verbose else 0,
    )
    XGB_model.fit(
        train_recs,
        train_labs,
        group=groups,
        verbose=False
    )
    _, _, ev_map, _, _ = evaluator(data_train, train_recs, XGB_model, training_dataframe, data_val)
    
    return ev_map

study.optimize(objective, n_trials=1000)
