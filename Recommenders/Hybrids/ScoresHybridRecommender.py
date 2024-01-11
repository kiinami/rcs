from Recommenders.BaseRecommender import BaseRecommender
import scipy.sparse as sps

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