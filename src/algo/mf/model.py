from typing import Any, Dict

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.algo.mf.dataset import UserItemRatingDFDataset

class MatrixFactorizationRating(nn.Module):
    """
    A matrix factorization model for rating prediction using embeddings for users and items.
    To predict ratings, the model computes the dot product of user and item embeddings.
    """

    def __init__(self, num_users: int, num_items: int, embedding_dim: int):
        """
        Initialize the model with user and item counts and the embedding dimension.

        Args:
            num_users (int): Number of users.
            num_items (int): Number of items.
            embedding_dim (int): Dimension of the embeddings.
        """
        super(MatrixFactorizationRating, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the predicted ratings.

        Args:
            user (torch.Tensor): User indices.
            item (torch.Tensor): Item indices.

        Returns:
            torch.Tensor: Predicted ratings.
        """
        user_emb = self.user_embedding(user)   # (batch_size, embedding_dim)
        item_emb = self.item_embedding(item)   # (batch_size, embedding_dim)
        
        output = torch.sum(user_emb * item_emb, dim=-1) # (batch_size,)
        return output
    
    def predict(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """
        Predict ratings for given user-item pairs.

        Args:
            users (torch.Tensor): User indices.
            items (torch.Tensor): Item indices.

        Returns:
            torch.Tensor: Predicted interactions scores.
        """
        output_ratings = self.forward(users, items)

        return nn.Sigmoid()(output_ratings)
    
    def predict_train_batch(
        self, batch_input: Dict[str, Any], device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """
        Predict scores for a batch of training data.

        Args:
            batch_input (Dict[str, Any]): A dictionary containing tensors with 'user' and 'item' indices.
            device (torch.device, optional): The device on which the model will run (CPU by default).

        Returns:
            torch.Tensor: The predicted scores for the batch.
        """
        users = batch_input["user"].to(device)
        items = batch_input["item"].to(device)
        return self.forward(users, items)
    
    def recommend(
            self, users: torch.Tensor, top_k: int = 10, batch_size: int = 1000
    ) -> Dict[int, Any]:
        """
        Recommend top-k items for each user.

        Args:
            users (torch.Tensor): User indices.
            top_k (int): Number of top items to recommend.
            batch_size (int): Size of each batch for prediction.

        Returns:
            Dict[int, Any]: A dictionary mapping user indices to recommended item indices.
        """
        self.eval()
        all_items = torch.arange(
            self.item_embedding.num_embeddings, device=users.device
        )

        user_indices = []
        recommendations = []
        scores = []

        with torch.no_grad():
            total_users = users.size(0)
            for i in tqdm(
                range(0, total_users, batch_size), desc="Generating recommendations"
            ):
                user_batch = users[i : i + batch_size]  # (batch_size,)

                # Expand user_batch to match all items
                user_batch_expanded = (
                    user_batch.unsqueeze(1).expand(-1, len(all_items)).reshape(-1)   # (batch_size,) -> (batch_size, 1) -> (batch_size, num_items) -> (batch_size * num_items,)
                )  

                items_batch = (
                    all_items.unsqueeze(0).expand(len(user_batch), -1).reshape(-1)   # (num_item,) -> (1, num_items) -> (batch_size, num_items) -> (batch_size * num_items,)
                ) 

                # Predict scores for the batch
                batch_scores = self.predict(user_batch_expanded, items_batch).view(
                    len(user_batch), -1
                )  # (batch_size, num_items)

                # Get top k items for each user in the batch
                topk_scores, topk_indices = torch.topk(batch_scores, top_k, dim=1)
                topk_items = all_items[topk_indices]  # (batch_size, top_k)

                # Collect recommendations
                user_indices.extend(user_batch.repeat_interleave(top_k).cpu().tolist())
                recommendations.extend(topk_items.cpu().flatten().tolist())   
                scores.extend(topk_scores.cpu().flatten().tolist())   

        return {
            "user_indice": user_indices,
            "recommendation": recommendations,
            "score": scores,
        }
    
    @classmethod
    def get_default_dataset(cls):
        """
        Returns the default dataset class for the model.
        """
        return UserItemRatingDFDataset

    


# users = torch.tensor([i for i in range(32)])
# user_batch = users[0:16]
# print(user_batch.unsqueeze(1).expand(-1, 10).reshape(-1))\


# user_batch = torch.tensor([0, 1, 2])
# user_batch.repeat_interleave(3)
# >> tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])

# To do : add a test in a separate file
mf = MatrixFactorizationRating(5, 10, 128)
# print(mf.forward(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])))
# print(mf.predict(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])))