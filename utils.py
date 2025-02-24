"""
utils.py
2023-11-02 kiinami
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import numpy as np
from random import randint
import csv
from Recommenders.NonPersonalizedRecommender import TopPop


def get_database_url():
    if not os.path.isfile('data/database_url.txt'):
        return None
    with open('data/database_url.txt', 'r') as f:
        return f.read()


def load_data():
    return pd.read_csv('data/data_train.csv',
                       sep=',',
                       names=['user_id', 'item_id', 'rating'],
                       header=0,
                       dtype={'user_id': int,
                              'item_id': int,
                              'rating': np.double})


def load_users():
    return pd.read_csv('data/data_target_users_test.csv',
                       sep=',',
                       names=['user_id'],
                       header=0,
                       dtype={'user_id': int})['user_id'].to_numpy()


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

    precision_at_k = is_relevant * np.cumsum(is_relevant, dtype=float) / (1 + np.arange(is_relevant.shape[0]))

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

        recommendations = np.array(recommender.recommend(user_id,
                                                cutoff=recommendation_length))

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

        recommendations = np.array(recommender.recommend(mapped_user_id,
                                                cutoff=recommendation_length))

        submission[user_id] = [mapping_to_item_id[item_id] for item_id in recommendations]
    
    for user_id in users_to_recommend:
        if user_id not in submission:
            submission[user_id] = np.random.choice(item_ids, 10)

    return submission


def write_submission(submissions):
    with open('data/results.csv', "w+") as f:
        f.write("user_id,item_list\n")
        for user_id, items in submissions.items():
            f.write(f"{user_id},{' '.join([str(item) for item in items])}\n")


# New version of the functions


def load_data2():
    with open('data/data_train.csv', newline='') as f:
        reader = csv.reader(f)
        reader.__next__()
        data_raw = [(int(u), int(i), float(r)) for u, i, r in reader]
    
    usermap = {}
    for u, _, _ in data_raw:
        if u not in usermap:
            usermap[u] = len(usermap)
            
    itemmap = {}
    for _, i, _ in data_raw:
        if i not in itemmap:
            itemmap[i] = len(itemmap)
            
    data = [(usermap[u], itemmap[i], r) for u, i, r in data_raw]
    
    
    with open('data/data_target_users_test.csv', newline='') as f:
        reader = csv.reader(f)
        reader.__next__()
        users = [int(u) for u, in reader]
        for u in users:
            if u not in usermap:
                usermap[u] = len(usermap)
        users = [usermap[u] for u in users]
    
    return data, {v: k for k, v in usermap.items()}, {v: k for k, v in itemmap.items()}, users


def split_data2(data, testing_percentage: float, validation_percentage: float = None, seed: int = None):
    num_users = len(set([u for u, _, _ in data]))
    num_items = len(set([i for _, i, _ in data]))
    
    if seed is None:
        seed = randint(0, 10000)
    
    (user_ids_training, user_ids_test,
     item_ids_training, item_ids_test,
     ratings_training, ratings_test) = train_test_split([u for u, _, _ in data],
                                                        [i for _, i, _ in data],
                                                        [r for _, _, r in data],
                                                        test_size=testing_percentage,
                                                        shuffle=True,
                                                        random_state=seed)

    if not validation_percentage:
        return sp.csr_matrix((ratings_training, (user_ids_training, item_ids_training)),
                              shape=(num_users, num_items)), \
               sp.csr_matrix((ratings_test, (user_ids_test, item_ids_test)),
                             shape=(num_users, num_items))

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


def submission2(recommender, users, usermap, itemmap, data_train):
    
    toppoprecommender = TopPop(data_train)
    toppoprecommender.fit()
    
    with open('data/results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'item_list'])
        for user_id in users:
            try:
                item_list = recommender.recommend(user_id, cutoff=10, remove_seen_flag=True, remove_top_pop_flag=True)
                item_list = [itemmap[i] for i in item_list]
                writer.writerow([usermap[user_id], ' '.join(map(str, item_list))])
            except IndexError:
                item_list = toppoprecommender.recommend(0, cutoff=10, remove_seen_flag=True)
                item_list = [itemmap[i] for i in item_list]
                writer.writerow([usermap[user_id], ' '.join(map(str, item_list))])


from Recommenders.BaseRecommender import BaseRecommender
import scipy.sparse as sps
from numpy import linalg as LA
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


    def _compute_item_score(self, user_id_array, items_to_compute):
        
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)

        norm_item_weights_1 = LA.norm(item_weights_1, self.norm)
        norm_item_weights_2 = LA.norm(item_weights_2, self.norm)
        
        
        if norm_item_weights_1 == 0:
            norm_item_weights_1 = 0.0000001
            #raise ValueError("Norm {} of item weights for recommender 1 is zero. Avoiding division by zero".format(self.norm))
        
        if norm_item_weights_2 == 0:
            norm_item_weights_2 = 0.0000001
            #raise ValueError("Norm {} of item weights for recommender 2 is zero. Avoiding division by zero".format(self.norm))
        
        item_weights = item_weights_1 / norm_item_weights_1 * self.alpha + item_weights_2 / norm_item_weights_2 * (1-self.alpha)

        return item_weights