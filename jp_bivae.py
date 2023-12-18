import os
from utils import to_csr, precision, recall, mean_average_precision, get_database_url
import optuna
from Recommenders.BiVAE.dataset import Dataset
from Recommenders.BiVAE.BiVAECF import BiVAECF
import csv
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import numpy as np


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
    
    return data, usermap, itemmap, users


def split_data2(data, testing_percentage: float, validation_percentage: float):
    train_percentage = 1 - testing_percentage - validation_percentage
    train_data, temp_data = train_test_split(data, test_size=1-train_percentage)
    if testing_percentage == 0:
        return train_data, None, temp_data
    test_data, val_data = train_test_split(temp_data, test_size=validation_percentage/(testing_percentage + validation_percentage))
        
    return to_csr(train_data), to_csr(test_data), to_csr(val_data)


def evaluator(recommender: object, data_train, data_test: sp.csr_matrix, recommendation_length: int = 10):
    accum_precision = 0
    accum_recall = 0
    accum_map = 0

    num_users = len(data_train.user_ids)

    num_users_evaluated = 0
    num_users_skipped = 0
    for user_id in range(num_users):
        user_profile_start = data_test.indptr[user_id]
        user_profile_end = data_test.indptr[user_id+1]

        relevant_items = data_test.indices[user_profile_start:user_profile_end]

        if relevant_items.size == 0:
            num_users_skipped += 1
            continue
        
        if not usermap_reverse.get(user_id):
            num_users_skipped += 1
            continue

        recommendations = np.array(recommender.recommend(usermap_reverse[user_id], k=recommendation_length))

        accum_precision += precision(recommendations, relevant_items)
        accum_recall += recall(recommendations, relevant_items)
        accum_map += mean_average_precision(recommendations, relevant_items)

        num_users_evaluated += 1


    accum_precision /= max(num_users_evaluated, 1)
    accum_recall /= max(num_users_evaluated, 1)
    accum_map /=  max(num_users_evaluated, 1)

    return accum_precision, accum_recall, accum_map, num_users_evaluated, num_users_skipped


data, usermap, itemmap, users = load_data2()
usermap_reverse = {v: k for k, v in usermap.items()}
itemmap_reverse = {v: k for k, v in itemmap.items()}
data_train, data_test, data_val = split_data2(data, 0, 0.2)

data_train = Dataset.build(data_train, global_uid_map=usermap, global_iid_map=itemmap)
data_val = Dataset.build(data_val, global_uid_map=usermap, global_iid_map=itemmap)


study_name = "BiVAE"
study = optuna.create_study(
    study_name=study_name,
    storage=get_database_url(),
    load_if_exists=True,
    direction="maximize",
)



def objective(trial):
    encoder_structure = trial.suggest_categorical('encoder_structure', [20, 40, 60, 80, 100, 200, 300])
    act_fn = trial.suggest_categorical('act_fn', ['sigmoid', 'tanh', 'elu', 'relu', 'relu6'])
    likelihood = trial.suggest_categorical('likelihood', ['pois', 'gaus', 'bern'])
    epochs = trial.suggest_int('epochs', 10, 150)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512, 1024, 2048, 4096])
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-1, log=True)

    recommender = BiVAECF(
        k=10,
        encoder_structure=[encoder_structure],
        act_fn=act_fn,
        likelihood=likelihood,
        n_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        verbose=True,
        use_gpu=True,
    )
    recommender.fit(data_train)
    _, _, ev_map, _, _ = evaluator(recommender, data_train, data_val.matrix)
    
    return ev_map


study.optimize(objective, n_trials=150)
