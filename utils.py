"""
utils.py
2023-11-02 kiinami
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import numpy as np
from random import randint


def load_data():
    return pd.read_csv('../data/data_train.csv',
                       sep=',',
                       names=['user_id', 'item_id', 'rating'],
                       header=0,
                       dtype={'user_id': np.int32,
                              'item_id': np.int32,
                              'rating': np.double})


def load_users():
    return pd.read_csv('../data/data_target_users_test.csv',
                       sep=',',
                       names=['user_id'],
                       header=0,
                       dtype={'user_id': np.int32})['user_id'].to_numpy()


def preprocess_data(data: pd.DataFrame):
    unique_users = data.user_id.unique()
    unique_items = data.item_id.unique()

    mapping_user_id = pd.DataFrame({"mapped_user_id": np.arange(unique_users.size), "user_id": unique_users})
    mapping_item_id = pd.DataFrame({"mapped_item_id": np.arange(unique_items.size), "item_id": unique_items})

    data = pd.merge(left=data,
                    right=mapping_user_id,
                    how="inner",
                    on="user_id")

    data = pd.merge(left=data,
                    right=mapping_item_id,
                    how="inner",
                    on="item_id")

    return data, unique_users.size, unique_items.size, mapping_user_id


def split_data(ratings, num_users, num_items, validation_percentage: float, testing_percentage: float):
    seed = randint(0, 10000)

    (user_ids_training, user_ids_test,
     item_ids_training, item_ids_test,
     ratings_training, ratings_test) = train_test_split(ratings.mapped_user_id,
                                                        ratings.mapped_item_id,
                                                        ratings.rating,
                                                        test_size=testing_percentage,
                                                        shuffle=True,
                                                        random_state=seed)

    (user_ids_training, user_ids_validation,
     item_ids_training, item_ids_validation,
     ratings_training, ratings_validation) = train_test_split(user_ids_training,
                                                              item_ids_training,
                                                              ratings_training,
                                                              test_size=validation_percentage,
                                                              )

    urm_train = sp.csr_matrix((ratings_training, (user_ids_training, item_ids_training)),
                              shape=(num_users, num_items))

    urm_validation = sp.csr_matrix((ratings_validation, (user_ids_validation, item_ids_validation)),
                                   shape=(num_users, num_items))

    urm_test = sp.csr_matrix((ratings_test, (user_ids_test, item_ids_test)),
                             shape=(num_users, num_items))

    return urm_train, urm_validation, urm_test


def recall(recommendations: np.array, relevant_items: np.array) -> float:
    is_relevant = np.in1d(recommendations, relevant_items, assume_unique=True)

    recall_score = np.sum(is_relevant) / relevant_items.shape[0]

    return recall_score


def precision(recommendations: np.array, relevant_items: np.array) -> float:
    is_relevant = np.in1d(recommendations, relevant_items, assume_unique=True)

    precision_score = np.sum(is_relevant) / recommendations.shape[0]

    return precision_score

def mean_average_precision(recommendations: np.array, relevant_items: np.array) -> float:
    is_relevant = np.in1d(recommendations, relevant_items, assume_unique=True)

    precision_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

    map_score = np.sum(precision_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score


def calculate_metrics(recommendations: np.array, relevant_items: np.array):
    recall_score = recall(recommendations, relevant_items)
    precision_score = precision(recommendations, relevant_items)
    map_score = mean_average_precision(recommendations, relevant_items)

    return recall_score, precision_score, map_score


def evaluator(recommender: object, data_train: sp.csr_matrix, data_test: sp.csr_matrix, recommendation_length: int = 10):
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

        recommendations = recommender.recommend(user_id=user_id,
                                                at=recommendation_length,
                                                urm_train=data_train,
                                                remove_seen=True)

        accum_precision += precision(recommendations, relevant_items)
        accum_recall += recall(recommendations, relevant_items)
        accum_map += mean_average_precision(recommendations, relevant_items)

        num_users_evaluated += 1


    accum_precision /= max(num_users_evaluated, 1)
    accum_recall /= max(num_users_evaluated, 1)
    accum_map /=  max(num_users_evaluated, 1)

    return accum_precision, accum_recall, accum_map, num_users_evaluated, num_users_skipped


def prepare_submission(ratings: pd.DataFrame, users_to_recommend: np.array, urm_train: sp.csr_matrix, recommender: object):
    users_ids_and_mappings = ratings[ratings.user_id.isin(users_to_recommend)][["user_id", "mapped_user_id"]].drop_duplicates()

    mapping_to_item_id = dict(zip(ratings.mapped_item_id, ratings.item_id))
    item_ids = ratings.item_id.unique()


    recommendation_length = 10
    submission = dict()
    for idx, row in users_ids_and_mappings.iterrows():
        user_id = row.user_id
        mapped_user_id = row.mapped_user_id

        recommendations = recommender.recommend(user_id=mapped_user_id,
                                                urm_train=urm_train,
                                                at=recommendation_length,
                                                remove_seen=True)

        submission[user_id] = [mapping_to_item_id[item_id] for item_id in recommendations]
    
    for user_id in users_to_recommend:
        if user_id not in submission:
            submission[user_id] = np.random.choice(item_ids, 10)

    return submission


def write_submission(submissions):
    with open('../data/results.csv', "w+") as f:
        f.write("user_id,item_list\n")
        for user_id, items in submissions.items():
            f.write(f"{user_id},{' '.join([str(item) for item in items])}\n")
