from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.algo.two_tower.dataset import UserItemRatingDFDataset

class TwoTowerRating(nn.Module):
    """
    A matrix factorization model for rating prediction using embeddings for users and items.
    To predict ratings, the model computes the dot product of user and item embeddings.
    """

    def __init__(self, num_users: int, num_items: int, embedding_dim: int, hidden_units_dim: int, dropout: float = 0.2):
        """
        Initialize the model with user and item counts and the embedding dimension.

        Args:
            num_users (int): Number of users.
            num_items (int): Number of items.
            embedding_dim (int): Dimension of the embeddings.
            hidden_units_dim (int): Dimension of the hidden units.
            dropout (float): Dropout rate for regularization.
        """
        super(TwoTowerRating, self).__init__()
        
        # Itemn tower
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.item_fc = nn.Linear(embedding_dim, hidden_units_dim)

        # User tower
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.user_fc = nn.Linear(embedding_dim, hidden_units_dim)
        
        # Logits
        # self.output_fc = nn.Linear(hidden_units_dim *2, 1)  # (batch_size, hidden_units_dim) -> (batch_size, 1)
        
        # Activation and dropout layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid_fn = nn.Sigmoid()

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the predicted ratings.

        Args:
            user (torch.Tensor): User indices.
            item (torch.Tensor): Item indices.

        Returns:
            torch.Tensor: Predicted ratings.
        """
        # Query tower
        query_emb = self._get_query_tower(user)  # (batch_size, hidden_units_dim)
        # query_emb = F.normalize(query_emb, p=2, dim=-1)  # Normalize the user embedding
        
        # Candidate tower
        candidate_emb = self._get_candidate_tower(item)  # (batch_size, hidden_units_dim)
        # candidate_emb = F.normalize(candidate_emb, p=2, dim=-1)  # Normalize the item embedding
        
        # # Output layer
        # combined = torch.cat([query_emb, candidate_emb], dim=-1)  # (batch_size, hidden_units_dim * 2)
        # output = self.output_fc(combined)
        
        # Compute the dot product of user and item embeddings
        # output = torch.sum(query_emb * candidate_emb, dim=-1)
        output = F.cosine_similarity(query_emb, candidate_emb, dim=-1)  # (batch_size,)
        # output = self.sigmoid_fn(output)  # Apply sigmoid activation to the output
        # return (output+1)/2
        return output
    
    def _get_query_tower(self, user: torch.Tensor) -> torch.Tensor:
        
        """
        Get the query tower output for a given user.

        Args:
            user (torch.Tensor): User indices. # (batch_size,)

        Returns:
            torch.Tensor: Query tower output.
        """
        user_emb = self.user_embedding(user) # (batch_size, embedding_dim)
        
        user_x = user_emb
        
        # user_x = self.user_fc(user_emb)   # (batch_size, hidden_units_dim)
        # user_x = self.relu(user_x)   # (batch_size, hidden_units_dim)
        # user_x = self.dropout(user_x)
        return user_x
    
    def _get_candidate_tower(self, item: torch.Tensor) -> torch.Tensor:
        """
        Get the candidate tower output for a given item.

        Args:
            item (torch.Tensor): Item indices. # (batch_size,)

        Returns:
            torch.Tensor: Candidate tower output.
        """
        item_emb = self.item_embedding(item)
        
        item_x = item_emb
        
        # item_x = self.item_fc(item_emb)   # (batch_size, hidden_units_dim)
        # item_x = self.relu(item_x)   # (batch_size, hidden_units_dim)
        # item_x = self.dropout(item_x)
        return item_x
        
    
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
        output_ratings = self.sigmoid_fn(output_ratings)  # Apply sigmoid activation to the output

        return output_ratings
    
    
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
# tt = TwoTowerRating(5, 10, 128, 128)
# print(tt.forward(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])))
# print(tt.predict(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])))
# print(tt.recommend(torch.tensor([0, 1, 2]), top_k=2))