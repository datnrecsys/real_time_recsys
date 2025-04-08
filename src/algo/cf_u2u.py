import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional
import torch
from loguru import logger
from typing import Union, Dict, Any
from tqdm import tqdm
from src.utils.math_utils import sigmoid

# def sigmoid(z: Union[float, int, np.ndarray, list]) -> Union[np.float64, np.ndarray]:
#     return 1 / (1 + np.exp(-z))

class U2UCollaborativeFiltering:
    '''
    This class implement u2u collaborative filtering algorithm. Basically, this algorithm is based on the user's history in order to
    find the most similar users to the current(target) user. Then, the algorithm will recommend items that the most similar users have
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

        # Construct user-user similarity matrix, with values are the cosine similarity between users and the shape is (n_users, n_users)
        self.u2u_matrix = np.zeros((n_users, n_users))

    def forward(self, user_idx:int, item_idx:int, top_n: Optional[int] = 10, logging: Optional[bool] = False) -> int:
        '''
        Predict the rating of the user to the item.
        Args:
            user_idx: int, the index of the user
            item_idx: int, the index of the item
            top_n: int, the number of most similar users to consider
        Returns:
            rating: float, the predicted rating of the user to the item
        '''
        
        # Get the similarity between the current user and all other users. Shape: (n_users,)
        user_sim = self.u2u_matrix[user_idx]

        # Get the rating of all users for the current item. Shape: (n_users,)
        item_rating = self.user_item_matrix[:, item_idx]
        
        # Get only the users who have rated the item
        rated_mask = item_rating != 0
        user_sim = user_sim[rated_mask]
        item_rating = item_rating[rated_mask]

        if len(item_rating) == 0 :
            if logging:
                logger.info(f"Item {item_idx} has no ratings. Return 0 instead.")
            return 0
        
        if sum(user_sim) == 0:
            if logging:
                logger.info(f"Users who rated the item {item_idx} share nothing in common with user {user_idx}. Return 0 instead.")
            return 0
        
        # Get the top-n most similar users with the current user
        top_k_users = np.argsort(user_sim)[-top_n:][::-1]

        user_sim = user_sim[top_k_users]
        item_rating = item_rating[top_k_users]
        # print("User sim", user_sim)
        # print("Item rating", item_rating)

        # Predict the rating of the user to the item
        # To do: Find the right way to calculate the weighted predicted rating
        logit = np.dot(user_sim, item_rating) / np.sum(user_sim)
        # print("Logit", logit)
        if logging:
            logger.debug(f"User sim: {user_sim}")
            logger.debug(f"Item rating: {item_rating}")
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

        # Calculate the user-user similarity matrix. Shape: (n_users, n_users)
        self.u2u_matrix = cosine_similarity(self.user_item_matrix)

        # Zero out self-similarity
        np.fill_diagonal(self.u2u_matrix, 0)

        assert self.u2u_matrix.shape == (self.n_users, self.n_users), "u2u_matrix shape mismatch"
    
    def predict(self, user_ids: Union[np.ndarray, list], item_ids: Union[np.ndarray, list], top_n: Optional[int] = 10, logging: Optional[bool] = False) -> np.ndarray:
        '''
        Predict the ratings of the users to the items.
        Args:
            user_ids: list, the list of user indices
            item_ids: list, the list of item indices
            top_n: int, the number of most similar users to consider
        Returns:
            ratings: np.ndarray, the array of predicted ratings
        '''


        user_ids = np.asarray(user_ids)
        item_ids = np.asarray(item_ids)

        predictions = np.array(
            [self.forward(user_idx, item_idx, top_n, logging) for user_idx, item_idx in zip(user_ids, item_ids)]
        )
        return sigmoid(predictions)
    
    def recommend(self, user_ids: Union[np.ndarray, list], top_n: Optional[int] = 10, k: Optional[int] = 100, keep_interacted: Optional[bool] = False) -> Dict[str, Any]:
        '''
        Recommend the top-n items for the users.
        Args:
            user_ids: list, the list of user indices
            k: int, the number of items to recommend
            top_n: int, the number of most similar users to consider
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
            predictions = self.predict([user_id] * len(unseen_items), unseen_items, top_n)

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

# Put a dummy test here
# To do: Write test for this class
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

# cf = U2UCollaborativeFiltering(n_users, n_items)
# cf.fit(rating_test["user_id"], rating_test["item_id"], rating_test["rating"])
# predicted_rating = cf.predict([1], [1])
# print(predicted_rating)
# rec = cf.recommend([0], k=10, keep_interacted=True)
# print(rec)