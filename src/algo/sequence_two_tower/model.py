from typing import Any, Dict, Optional, cast

import torch
import torch.nn as nn
from loguru import logger
from torch.nn import functional as F
from tqdm.auto import tqdm

from src.algo.base.base_dl_model import BaseDLModel
from src.algo.sequence_two_tower.dataset import UserItemBinaryRatingDFDataset
from src.domain.model_request import SequenceModelRequest


class SequenceRatingPrediction(BaseDLModel):
    """
    A PyTorch neural network model for predicting user-item interaction ratings based on sequences of previous items
    and a target item. This model uses user and item embeddings, and performs rating predictions using fully connected layers.

    Args:
        num_users (int): The number of unique users.
        num_items (int): The number of unique items.
        embedding_dim (int): The size of the embedding dimension for both user and item embeddings.
        item_embedding (torch.nn.Embedding): pretrained item embeddings. Defaults to None.
        dropout (float, optional): The dropout probability applied to the fully connected layers. Defaults to 0.2.

    Attributes:
        num_items (int): Number of unique items.
        num_users (int): Number of unique users.
        item_embedding (nn.Embedding): Embedding layer for items, including a padding index for unknown items.
        user_embedding (nn.Embedding): Embedding layer for users.
        fc_rating (nn.Sequential): Fully connected layers for predicting the rating from concatenated embeddings.
        relu (nn.ReLU): ReLU activation function.
        dropout (nn.Dropout): Dropout layer applied after activation.
    """

    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim,
        item_embedding: Optional[torch.nn.Embedding] = None,
        dropout=0.2,
        use_user_embedding: bool = True,
        use_start_token: bool = False,
    ):
        super().__init__()

        self.num_items = num_items
        self.num_users = num_users

        # Item embedding (with padding index and <start> index)
        if use_start_token:
            self.item_embedding = nn.Embedding(
                num_items + 2,
                embedding_dim,
                padding_idx=num_items+1
            )
        else:
            self.item_embedding = nn.Embedding(
                num_items + 1,
                embedding_dim,
                padding_idx=num_items
            )
        
        if item_embedding:
            self.item_embedding.weight.data[:-2] = item_embedding.weight.data if item_embedding.num_embeddings == num_items else item_embedding.weight.data[:-1]
            

        # GRU layer to process item sequences
        # self.gru = nn.GRU(
        #     input_size=embedding_dim,
        #     hidden_size=embedding_dim,
        #     batch_first=True
        # )
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=embedding_dim,
            batch_first=True,
            dropout=dropout,
            activation=nn.PReLU(),
            layer_norm_eps=1e-5,
        )
        
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=1,
        )

        # self.gelu = nn.ReLU()
        self.prelu1 = nn.PReLU()
        # self.prelu2 = nn.PReLU()
        self.dropout = nn.Dropout(p=dropout)
        
        if use_user_embedding:
            # User embedding
            self.user_embedding = nn.Embedding(num_users, embedding_dim)
            self.final_user_fc = nn.Linear(embedding_dim * 2, embedding_dim)
        else:
            self.final_fc = nn.Linear(embedding_dim * 2, embedding_dim)
        # self.final_fc2 = nn.Linear(embedding_dim, embedding_dim // 2)
        self.score_fc = nn.Linear(embedding_dim, 1)
        
        self._use_user_embedding = use_user_embedding
        self._use_start_token = use_start_token

        # Fully connected layers for rating prediction
        # self.query_fc = nn.Sequential(
        #     nn.Linear(embedding_dim, embedding_dim),
        #     # self.gelu,
        #     # nn.BatchNorm1d(embedding_dim),
        #     self.gelu,
        # )
        
        # self.candidate_fc = nn.Sequential(
        #     nn.Linear(embedding_dim, embedding_dim ),
        #     # nn.BatchNorm1d(embedding_dim),
        #     self.gelu,
        #     # self.gelu,
            
        # )
        logger.info(f"Start token used: {self._get_item_start_token_idx}"
                    f", Padding token used: {self._get_item_padding_token_idx}")
        

    def forward(self, input_data: SequenceModelRequest)-> torch.Tensor:
        """
        Forward pass to predict the rating.

        Args:
            user_ids (torch.Tensor): Batch of user IDs.
            input_seq (torch.Tensor): Batch of item sequences.
            target_item (torch.Tensor): Batch of target items to predict the rating for.

        Returns:
            torch.Tensor: Predicted rating for each user-item pair.
        """
        if input_data.recommendation:
            raise ValueError(
                "Please set recommendation to False when calling forward method."
            )
        user_ids = input_data.user_id
        input_seq = input_data.item_sequence
        target_item = input_data.target_item

        query_embedding = self._get_user_tower(
            item_sequence=input_seq, user_indices=user_ids
        ) # Shape: [batch_size, embedding_dim]
        
        candidate_embedding = self._get_item_tower(target_item) # Shape: [batch_size, embedding_dim]
        
        query_embedding = F.normalize(query_embedding, p=2, dim=1)
        candidate_embedding = F.normalize(candidate_embedding, p=2, dim=1)
        
        cos_sim = torch.sum(query_embedding * candidate_embedding, dim=1)
        
        # Scale cosine similarity from [-1, 1] to [0, 1]
        return (cos_sim + 1) / 2
    
    def predict(
        self, input_data: SequenceModelRequest
    ):
        output_ratings = self.forward(input_data)
        return output_ratings
    
    
    
    def _replace_negative_one_with_padding_idx(self, tensor: torch.Tensor) -> torch.Tensor:
        # Replace -1 in input_seq and target_item with num_items (padding_idx)
        padding_idx_tensor = torch.tensor(self.item_embedding.padding_idx)
        new_tensor = torch.where(tensor == -1, padding_idx_tensor, tensor)
        
        return new_tensor
    
    @property
    def _get_item_start_token_idx(self) -> int:
        """
        Get the ID of the start token used in the item embedding.

        Returns:
            int: The ID of the start token.
        """
        return self.item_embedding.num_embeddings - 2
    
    @property
    def _get_item_padding_token_idx(self) -> int:
        """
        Get the ID of the padding token used in the item embedding.

        Returns:
            int: The ID of the padding token.
        """
        assert self.item_embedding.padding_idx == self.item_embedding.num_embeddings - 1, "Padding index should be the last index in the item embedding."
        return self.item_embedding.padding_idx

    def _get_user_tower(self, item_sequence: torch.Tensor, user_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get the user tower output for a given user.

        Args:
            user (torch.Tensor): User indices. # (batch_size,)

        Returns:
            torch.Tensor: User tower output.
        """
        
        if self._use_user_embedding:
            assert user_indices is not None, "user_indices must be provided when using user embedding."
            user_emb = self.user_embedding(user_indices) # Shape: [batch_size, embedding_dim]
        
        item_sequence = self._replace_negative_one_with_padding_idx(item_sequence)
        
        # Pad start token at the beginning of the sequence
        if self._use_start_token:
            item_sequence = F.pad(item_sequence, (0, 1), mode='constant', value=self._get_item_start_token_idx)

        mask = (item_sequence == self._get_item_padding_token_idx).float()
        # print("mask: ", mask)
        # print(mask.shape) #(batch_size, seq_len)

        # Embed input sequence
        embedded_seq = self.item_embedding(
                item_sequence
            )  # Shape: [batch_size, seq_len, embedding_dim]
        
        hidden_state = self.encoder.forward(
            src = embedded_seq,
            src_key_padding_mask=mask,
        ) # Shape: [batch_size, seq_len, embedding_dim]
        
        hidden_state = hidden_state[:, -1, :]  # Shape: [batch_size, embedding_dim]
        
        if self._use_user_embedding:
            hidden_state = torch.cat(
                (hidden_state, user_emb), dim=-1
            )  # Shape: [batch_size, embedding_dim * 2]

            hidden_state = self.final_user_fc(hidden_state)  # Shape: [batch_size, embedding_dim]

        return hidden_state

    def _get_item_tower(self, item_indices: torch.Tensor) -> torch.Tensor:
        """
        Get the item tower output for a given item sequence.

        Args:
            item_indices (torch.Tensor): Item  tensor. # (batch_size,)

        Returns:
            torch.Tensor: Item tower output.
        """
        item_indices = self._replace_negative_one_with_padding_idx(item_indices)
        item_embedding = self.item_embedding(item_indices)  # Shape: [batch_size, embedding_dim]
        assert item_embedding.shape[-1] == self.item_embedding.embedding_dim, \
            f"Item embedding dimension mismatch: expected {self.item_embedding.embedding_dim}, got {item_embedding.shape[-1]}"
        
        return item_embedding
    
    @classmethod
    def get_default_dataset(cls):
        """
        Returns the expected dataset type for the model.

        Returns:
            UserItemRatingDFDataset: The expected dataset type.
        """
        return UserItemBinaryRatingDFDataset
    

    def recommend(
        self,
        input_data: SequenceModelRequest,
        k: int,
        batch_size: int = 128,
    ) -> Dict[str, Any]:
        """
        Generate top-k recommendations for a batch of users based on their item sequences.

        Args:
            users (torch.Tensor): Tensor containing user IDs.
            item_sequences (torch.Tensor): Tensor containing sequences of previously interacted items.
            k (int): Number of recommendations to generate for each user.
            batch_size (int, optional): Batch size for processing user-item pairs. Defaults to 128.

        Returns:
            Dict[str, Any]: Dictionary containing recommended items and scores:
                'user_indice': List of user indices.
                'recommendation': List of recommended item indices.
                'score': List of predicted interaction scores.
        """
        users = input_data.user_id
        item_sequences = input_data.item_sequence
        
        self.eval()
        all_items = torch.arange(
            self.item_embedding.num_embeddings, device=users.device
        )

        # Create all user-item pairs
        user_batch_expanded = users.unsqueeze(1).expand(-1, len(all_items)).reshape(-1)
        items_batch = all_items.unsqueeze(0).expand(len(users), -1).reshape(-1)
        item_sequences_batch = item_sequences.unsqueeze(1).repeat(1, len(all_items), 1)
        item_sequences_batch = item_sequences_batch.view(-1, item_sequences.size(-1))

        all_scores = []

        with torch.no_grad():
            total_pairs = len(user_batch_expanded)
            for i in tqdm(
                range(0, total_pairs, batch_size), desc="Generating recommendations"
            ):
                end_idx = min(i + batch_size, total_pairs)
                
                batch_users = user_batch_expanded[i:end_idx]
                batch_items = items_batch[i:end_idx]
                batch_sequences = item_sequences_batch[i:end_idx]
                
                input_data_batch = SequenceModelRequest(
                    user_id=batch_users,
                    item_sequence=batch_sequences,
                    target_item=batch_items,
                    recommendation=False
                )

                # Predict scores for the batch
                batch_scores = self.predict(input_data_batch)
                all_scores.append(batch_scores)

        # Concatenate all scores and reshape
        all_scores = torch.cat(all_scores, dim=0)
        all_scores = all_scores.view(len(users), len(all_items))

        # Get top k items for each user
        topk_scores, topk_indices = torch.topk(all_scores, k, dim=1)
        topk_items = all_items[topk_indices]

        # Collect recommendations
        user_indices = users.repeat_interleave(k).cpu().tolist()
        recommendations = topk_items.cpu().flatten().tolist()
        scores = topk_scores.cpu().flatten().tolist()

        return {
            "user_indice": user_indices,
            "recommendation": recommendations,
            "score": scores,
        }

    


# users = torch.tensor([i for i in range(32)])
# user_batch = users[0:16]
# print(user_batch.unsqueeze(1).expand(-1, 10).reshape(-1))\


# user_batch = torch.tensor([0, 1, 2])
# user_batch.repeat_interleave(3)
# >> tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])

# To do : add a test in a separate file
# seq = SequenceRatingPrediction(5, 10, 128)
# print(seq.forward(torch.tensor([0, 1, 2]), torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), torch.tensor([0, 1, 2])))
# print(seq.predict(torch.tensor([0, 1, 2]), torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), torch.tensor([0, 1, 2])))
# print(seq.recommend(torch.tensor([0, 1, 2]), torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), k=2))