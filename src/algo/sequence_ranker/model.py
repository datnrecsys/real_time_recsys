from typing import Any, Dict, Optional, cast
from loguru import logger 

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from src.algo.sequence_ranker.dataset import UserItemBinaryRatingDFDataset
from torch.nn import functional as F

class SequenceRatingPrediction(nn.Module):
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
        user_embedding_dim: Optional[int] = None,
        use_title: bool = False,
        title_embedding: Optional[torch.nn.Embedding] = None,
        title_embedding_dim: Optional[int] = None,
        title_fc_dim: Optional[int] = None,
        use_extra_item_embedding: bool = False,
        extra_item_embedding_dim: Optional[int] = None,
        
    ):
        super().__init__()

        self.num_items = num_items
        self.num_users = num_users
        self.use_title = use_title
        self.use_extra_item_embedding = use_extra_item_embedding

        # Item embedding (with padding index and <start> index)
        # The last index is reserved for padding, and the second last for start token
        # If use_start_token is True, the last two indices are reserved for start and padding tokens
        # If use_start_token is False, only the last index is reserved for padding token
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
        
        # if item_embedding is not None, assign the fisrt num_items weights to the item_embedding
        # no need to assign the last two weights, as they are reserved for start and padding tokens   
        if item_embedding is not None:
            self.item_embedding.weight.data[:num_items] = item_embedding.weight.data

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

        # Metadata embedding (now title embedding)
        self.use_title = use_title
        self.title_embedding = title_embedding
        self.title_embedding_dim = title_embedding_dim
        self.title_fc_dim = title_fc_dim

        use_title_now = (
            self.use_title and
            self.title_embedding is not None and
            self.title_embedding_dim is not None and
            self.title_fc_dim is not None
        )

        use_extra_item_embedding_now = (
            self.use_extra_item_embedding and
            extra_item_embedding_dim is not None
        )

        if use_title_now:
            self.title_fc = nn.Linear(title_embedding_dim, title_fc_dim)
            self.title_prelu = nn.PReLU()
            item_tower_dim = embedding_dim + title_fc_dim
        else:
            item_tower_dim = embedding_dim

        if use_extra_item_embedding_now:
            assert extra_item_embedding_dim is not None, "extra_item_embedding_dim must be provided if use_extra_item_embedding is True"
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=dropout)
            self.extra_item_feature_layer = nn.Sequential(
                nn.Linear(extra_item_embedding_dim, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                self.relu,
                self.dropout,
            )
            item_tower_dim += embedding_dim

        self.item_tower_fc = nn.Linear(item_tower_dim, embedding_dim)
        self.item_tower_prelu = nn.PReLU()
        
        if use_user_embedding:
            # User embedding
            assert user_embedding_dim is not None, "user_embedding_dim must be provided if use_user_embedding is True"
            self.user_embedding = nn.Embedding(num_users, user_embedding_dim)
            user_embedding_dim = user_embedding_dim + embedding_dim 
        else:
            user_embedding_dim = embedding_dim
        self.user_tower_fc = nn.Linear(user_embedding_dim, embedding_dim)
        self.user_tower_prelu = nn.PReLU()
        
        self.final_fc = nn.Linear(embedding_dim * 2, embedding_dim)
        self.prelu2 = nn.PReLU()
        self.score_fc = nn.Linear(embedding_dim, 1)
        
        self._use_user_embedding = use_user_embedding
        self._use_start_token = use_start_token

        logger.info(f"Start token used: {self._get_item_start_token_idx}"
                    f", Padding token used: {self._get_item_padding_token_idx}")
        

    def forward(self, user_ids, input_seq, target_item, target_title=None, target_extra_embedding=None):
        """
        Forward pass to predict the rating.

        Args:
            user_ids (torch.Tensor): Batch of user IDs.
            input_seq (torch.Tensor): Batch of item sequences.
            target_item (torch.Tensor): Batch of target items to predict the rating for.

        Returns:
            torch.Tensor: Predicted rating for each user-item pair.
        """
        
        # Replace -1 in input_seq and target_item with num_items (padding_idx)
        input_seq = self._replace_negative_one_with_padding_idx(input_seq)
        target_item = self._replace_negative_one_with_padding_idx(target_item)

        # Pad start token at the beginning of the sequence
        if self._use_start_token:
            input_seq = F.pad(input_seq, (0, 1), mode='constant', value=self._get_item_start_token_idx)

        mask = (input_seq == self._get_item_padding_token_idx).float()

        # Embed input sequence
        embedded_seq = self.item_embedding(input_seq)  # Shape: [batch_size, seq_len, embedding_dim]

        hidden_state = self.encoder.forward(src = embedded_seq,src_key_padding_mask=mask,) # Shape: [batch_size, seq_len, embedding_dim]
        # Get the last hidden state
        hidden_state = hidden_state[:, -1, :]  # Shape: [batch_size, embedding_dim]

        # Embed the target item
        embedded_target = self.item_embedding(target_item)  # Shape: [batch_size, 1, embedding_dim]

        use_title = (
        self.use_title 
        and self.title_embedding is not None 
        and target_title is not None
        )

        if use_title:
            embedded_title = self.title_embedding(target_title)
            embedded_title = self.title_fc(embedded_title)
            embedded_title = self.title_prelu(embedded_title)
            item_tower_embedding = torch.cat((embedded_target, embedded_title), dim=-1)
        else:
            item_tower_embedding = embedded_target

        # Handle extra_item_embedding if enabled
        if self.use_extra_item_embedding :
            assert target_extra_embedding is not None, "target_extra_embedding must be provided if use_extra_item_embedding is True"
            extra_embedded = self.extra_item_feature_layer(target_extra_embedding)
            item_tower_embedding = torch.cat((item_tower_embedding, extra_embedded), dim=-1)

        item_tower_embedding = self.item_tower_fc(item_tower_embedding)
        item_tower_embedding = self.item_tower_prelu(item_tower_embedding)

        if self._use_user_embedding:
            embedded_user = self._get_user_tower(user_ids)  # Shape: [batch_size, user_embedding_dim]
        else:
            embedded_user = torch.tensor([], device = self.item_embedding.weight.device)

        user_tower_embedding = torch.cat((hidden_state, embedded_user), dim=-1)
        user_tower_embedding = self.user_tower_fc(user_tower_embedding)
        user_tower_embedding = self.user_tower_prelu(user_tower_embedding)


        combined_embedding = torch.cat((user_tower_embedding, item_tower_embedding), dim=-1)
        final_embedding = self.final_fc(combined_embedding)
        final_embedding = self.prelu1(final_embedding)
        final_embedding = self.dropout(final_embedding)
        final_embedding = self.prelu2(final_embedding)

        output_ratings = self.score_fc(final_embedding)
        output_ratings = cast(torch.tensor, output_ratings)
        output_ratings = output_ratings.masked_fill(torch.isnan(output_ratings), 0)
        return output_ratings  # Shape: [batch_size]
    
    def _replace_negative_one_with_padding_idx(self, tensor: torch.Tensor) -> torch.Tensor:
        # Replace -1 in input_seq and target_item with num_items (padding_idx)
        padding_idx_tensor = torch.tensor(self.item_embedding.padding_idx, device=tensor.device, dtype=tensor.dtype)
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
    
    def _get_user_tower(self, user_idx: torch.Tensor) -> torch.Tensor:
        """
        Get the user tower output for a given user.

        Args:
            user (torch.Tensor): User indices. # (batch_size,)

        Returns:
            torch.Tensor: User tower output.
        """
        user_emb = self.user_embedding(user_idx)
        return user_emb

    def predict(self, user, item_sequence, target_item, target_title=None, target_extra_embedding=None):
        """
        Predict the rating for a specific user and item sequence using the forward method
        and applying a Sigmoid function to the output.

        Args:
            user (torch.Tensor): User ID.
            item_sequence (torch.Tensor): Sequence of previously interacted items.
            target_item (torch.Tensor): The target item to predict the rating for.

        Returns:
            torch.Tensor: Predicted rating after applying Sigmoid activation.
        """
        output_rating = self.forward(user, item_sequence, target_item, target_title=target_title, target_extra_embedding=target_extra_embedding)
        # Apply sigmoid activation to the output
        output_rating = nn.Sigmoid()(output_rating)
        return output_rating

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
        users: torch.Tensor,
        item_sequences: torch.Tensor,
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

                # Predict scores for the batch
                if self.use_title and self.title_embedding is not None and self.use_extra_item_embedding:
                    batch_scores = self.predict(batch_users, batch_sequences, batch_items, target_title=batch_items, target_extra_embedding=None)
                else:
                    batch_scores = self.predict(batch_users, batch_sequences, batch_items)
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
# seq = SequenceRatingPrediction(5, 10, embedding_dim=128, user_embedding_dim=128)
# print(seq.forward(torch.tensor([0, 1, 2]), torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), torch.tensor([0, 1, 2])))
# print(seq.predict(torch.tensor([0, 1, 2]), torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), torch.tensor([0, 1, 2])))
# print(seq.recommend(torch.tensor([0, 1, 2]), torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), k=2))