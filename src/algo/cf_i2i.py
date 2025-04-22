import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from loguru import logger
from typing import Optional, Union,Dict, Any, List

from src.utils.math_utils import sigmoid

# def sigmoid(z: Union[float, int, np.ndarray, List]) -> Union[np.float64, np.ndarray]:
#     return 1 / (1 + np.exp(-z))

class I2ICollaborativeFiltering:
    '''
    This class implements item-to-item collaborative filtering algorithm. This algorithm is based on the item's history in order to
    find the most similar items to the current(target) item. Then, the algorithm will recommend items that the most similar items have
    interacted with.
    '''
    def __init__(self, n_users, n_items):
        '''
        Initialize the class with the number of users and items in the dataset.
        Args:
            n_users: int, number of users in the dataset
            n_items: int, number of items in the dataset
        '''
        self.n_users = n_users
        self.n_items = n_items
        
        # Construct user-item matrix, with values are the ratings that the user gave to the item (0-5) and the shape is (n_users, n_items)
        self.user_item_matrix = np.zeros((n_users, n_items))

        # Construct item-item similarity matrix, with values are the cosine similarity between items and the shape is (n_items, n_items)
        self.i2i_matrix = np.zeros((n_items, n_items))

    def forward(self, user_idx:int, item_idx:int, top_n: int = 10, min_sim_count: int = 1, logging: bool = False) -> int:
        '''
        Predict the rating of the user to the item.
        Args:
            item_idx: int, the index of the item
            user_idx: int, the index of the user
            top_n: int, the number of most similar items to consider
        Returns:
            rating: float, the predicted rating of the user to the item
        '''
        
        # Get the similarity between the current item and all other items. Shape: (n_items,)
        item_sim = self.i2i_matrix[item_idx]

        # Get the rating of current user for all items. Shape: (n_items,)
        user_rating = self.user_item_matrix[user_idx, :]
        
        # Get only the users who have rated the item
        rated_mask = user_rating != 0
        item_sim = item_sim[rated_mask]
        user_rating = user_rating[rated_mask]

        if len(user_rating) == 0 :
            if logging:
                logger.debug(f"User {user_idx} has not rated any items. Return 0 instead.")
            return 0
        
        if sum(item_sim) == 0:
            if logging:
                logger.debug(f"Item {item_idx} has no similar items. Return 0 instead.")
            return 0
        
        # Get top n most similar items with the current item and their ratings (descending order)
        top_n_item_idx = np.argsort(item_sim)[-top_n:][::-1]
        
        item_sim = item_sim[top_n_item_idx]
        user_rating = user_rating[top_n_item_idx]

        item_sim_count = np.sum(item_sim > 0)

        if min_sim_count > top_n:
            logger.warning(f"min_sim_count {min_sim_count} is greater than top_n {top_n}. Setting min_sim_count to top_n.")
            min_sim_count = top_n


        if item_sim_count < min_sim_count:
            if logging:
                logger.debug(f"Item {item_idx} has not enough similar items. Return 0 instead.")
            return 0
        
        
        logit = np.dot(item_sim, user_rating) / np.sum(item_sim)
        # print("Logit", logit)
        if logging:
            logger.debug(f"Item sim: {item_sim}")
            logger.debug(f"User rating: {user_rating}")
            logger.debug(f"Logit: {logit}")
        return logit

    def fit(self, users: Union[np.ndarray, list], items: Union[np.ndarray, list], ratings: Union[np.ndarray, list]):
        '''
        Fit the model with the user-item interactions.
        Args:
            users: np.ndarray or list, the array or list of user indices
            items: np.ndarray or list, the array or list of item indices
            ratings: np.ndarray or list, the array or list of ratings
        '''

        users = np.asarray(users)
        items = np.asarray(items)
        ratings = np.asarray(ratings)

        self.user_item_matrix[users, items] = ratings

        # Calculate the item-item similarity matrix. Shape: (n_items, n_items)
        self.i2i_matrix = cosine_similarity(self.user_item_matrix.T)

        # Zero out self-similarity
        np.fill_diagonal(self.i2i_matrix, 0)

        assert self.i2i_matrix.shape == (self.n_items, self.n_items), "i2i_matrix shape mismatch"

    def predict(self, user_ids: Union[np.ndarray, list], item_ids: Union[np.ndarray, list], top_n: Optional[int] = 10, min_sim_count: int = 1,logging: Optional[bool] = False) -> np.ndarray:
        '''
        Predict the ratings of the users to the items.
        Args:
            user_ids: list, the list of user indices
            item_ids: list, the list of item indices
            top_n: int, the number of most similar items to consider
        Returns:
            ratings: np.ndarray, the array of predicted ratings
        '''


        user_ids = np.asarray(user_ids)
        item_ids = np.asarray(item_ids)

        predictions = np.array(
            [self.forward(user_idx, item_idx, top_n, min_sim_count, logging) for user_idx, item_idx in zip(user_ids, item_ids)]
        )
        return sigmoid(predictions)
    
    def recommend(self, user_ids: Union[np.ndarray, list], top_n: Optional[int] = 10, k: Optional[int] = 100, min_sim_count: Optional[int] = 1, keep_interacted: Optional[bool] = False) -> Dict[str, Any]:
        '''
        Recommend the top-n items for the users.
        Args:
            user_ids: list, the list of user indices
            k: int, the number of items to recommend
            top_n: int, the number of most similar items to consider
            min_sim_count: int, the minimum number of similar items required to make a prediction
            keep_interacted: bool, whether to keep recommending the items that the user has interacted with in the recommendations
        Returns:
            recommendations: dict, the dictionary of recommended items for each user
        '''
        user_ids = np.asarray(user_ids)
        
        user_indices = []
        recommendations = []
        scores = []

        for user_id in tqdm(user_ids, desc="Recommending items", unit="user"):
            # Get the items that the user has not interacted with
            if keep_interacted:
                unseen_items = np.arange(self.n_items)
            else:
                unseen_items = np.where(self.user_item_matrix[user_id] == 0)[0]

            # Predict the ratings of the user to the unseen items
            predictions = self.predict([user_id] * len(unseen_items), unseen_items, top_n, min_sim_count)

            # Get the top-n items to recommend
            top_items = unseen_items[np.argsort(predictions)[::-1]][:k]
            top_scores = predictions[np.argsort(predictions)[::-1]][:k]

            user_indices.extend([user_id] * len(top_items))
            recommendations.extend(top_items)
            scores.extend(top_scores)

        return {
            "user_indice": user_indices,
            "recommendation": recommendations,
            "score": scores,
        }

# # Put a dummy test here
# # To do: Write test for this class
# import pandas as pd

# rating_test = pd.DataFrame(
#     {
#         "user_id": [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        
#         "item_id": [0, 1, 1, 2, 0, 2, 3, 1, 2, 4, 1, 2, 3],
#         "rating": [1, 4, 5, 1, 5, 4, 5, 5, 2, 5, 5, 2, 3 ]
#     }
# )

# n_users = rating_test["user_id"].nunique()
# n_items = rating_test["item_id"].nunique()

# cf = I2ICollaborativeFiltering(n_users, n_items)
# cf.fit(rating_test["user_id"], rating_test["item_id"], rating_test["rating"])
# print(cf.user_item_matrix)
# print(cf.i2i_matrix)
# predicted_rating = cf.predict([0], [4], logging=True)
# print(predicted_rating)
# rec = cf.recommend([0], k=10, keep_interacted=True, min_sim_count= 2)
# print(rec)