from typing import Any, Dict, Optional, cast

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from src.algo.sequence.dataset import UserItemBinaryRatingDFDataset
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
    ):
        super().__init__()

        self.num_items = num_items
        self.num_users = num_users

        # Item embedding (with padding index and <start> index)
        self.item_embedding = nn.Embedding(
            num_items + 2,
            embedding_dim,
            padding_idx=num_items+1
        )
        
        if item_embedding:
            self.item_embedding.weight.data[:-2] = item_embedding.weight.data if item_embedding.num_embeddings == num_items else item_embedding.weight.data[:-1]
            
        # User embedding
        self.user_embedding = nn.Embedding(num_users, embedding_dim)

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
        self.prelu2 = nn.PReLU()
        self.dropout = nn.Dropout(p=dropout)
        
        self.final_fc = nn.Linear(embedding_dim * 2, embedding_dim)
        self.final_fc2 = nn.Linear(embedding_dim, embedding_dim // 2)
        self.score_fc = nn.Linear(embedding_dim // 2, 1)

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
        

    def forward(self, user_ids, input_seq, target_item):
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
        input_seq = F.pad(input_seq, (1, 0), mode='constant', value=self._get_item_start_token_idx())

        mask = (input_seq == self._get_item_padding_token_idx()).float()
        # print(mask)
        # print(mask.shape) #(batch_size, seq_len)

        # Embed input sequence
        embedded_seq = self.item_embedding(
            input_seq
        )  # Shape: [batch_size, seq_len, embedding_dim]

        # # # # GRU processing: output the hidden states and the final hidden state
        # # hidden_state = embedded_seq.mean(dim=1)  # Mean pooling over the sequence dimension
        # _, hidden_state = self.gru(
        #     embedded_seq
        # )
        # hidden_state = hidden_state.squeeze(
        #     0
        # )  # Remove the sequence dimension -> [batch_size, embedding_dim]
        # # hidden_state = hidden_state.squeeze(0)
        # # query_embedding = self.query_fc(hidden_state)  # Shape: [batch_size, embedding_dim]
        # print("hehe")
        hidden_state = self.encoder.forward(
            src = embedded_seq,
            src_key_padding_mask=mask,
        ) # Shape: [batch_size, seq_len, embedding_dim]
        # print("hehe")
        # Get the last hidden state
        hidden_state = hidden_state[:, -1, :]  # Shape: [batch_size, embedding_dim]
        
        # print(hidden_state.shape)


        # Embed the target item
        embedded_target = self.item_embedding(
            target_item
        )  # Shape: [batch_size, 1, embedding_dim]
        # print(embedded_target)
        # candidate_embedding = self.candidate_fc(embedded_target)  # Shape: [batch_size, embedding_dim]

        # query_embedding = F.normalize(hidden_state, p = 2, dim= -1)
        # candidate_embedding = F.normalize(embedded_target, p = 2, dim= -1)
        # score = torch.sum(
        #     query_embedding * candidate_embedding, dim=-1
        # )
        final_embedding = torch.cat(
            (hidden_state, embedded_target), dim=-1
        )   # Shape: [batch_size, embedding_dim * 2]
        
        
        final_embedding = self.final_fc(final_embedding) # Shape: [batch_size, embedding_dim]
        final_embedding = self.prelu1(final_embedding)   # Shape: [batch_size, embedding_dim//2]
        
        final_embedding = self.final_fc2(final_embedding) # Shape: [batch_size, embedding_dim//2]
        final_embedding = self.prelu2(final_embedding)   # Shape: [batch_size, embedding_dim//2]
        
        
        # Apply sigmoid activation to the output
        # output_ratings = nn.Sigmoid()(score)
        # output_ratings = self.output_fc(
        #     torch.cat((combined_embedding, embedded_target), dim=1)
        # )
        output_ratings = self.score_fc(final_embedding) 

        output_ratings = cast(torch.tensor, output_ratings)
        output_ratings = output_ratings.masked_fill(torch.isnan(output_ratings), 0)
        # print(output_ratings) # Shape: [batch_size, 1]
        return output_ratings  # Shape: [batch_size]
    
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

    def predict(self, user, item_sequence, target_item):
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
        output_rating = self.forward(user, item_sequence, target_item)
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
            batch_size (int, optional): Batch size for processing users. Defaults to 128.

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

        user_indices = []
        recommendations = []
        scores = []

        with torch.no_grad():
            total_users = users.size(0)
            for i in tqdm(
                range(0, total_users, batch_size), desc="Generating recommendations"
            ):
                user_batch = users[i : i + batch_size]
                item_sequence_batch = item_sequences[i : i + batch_size]

                # Expand user_batch to match all items
                user_batch_expanded = (
                    user_batch.unsqueeze(1).expand(-1, len(all_items)).reshape(-1)
                )
                items_batch = (
                    all_items.unsqueeze(0).expand(len(user_batch), -1).reshape(-1)
                )
                item_sequences_batch = item_sequence_batch.unsqueeze(1).repeat(
                    1, len(all_items), 1
                )
                item_sequences_batch = item_sequences_batch.view(
                    -1, item_sequence_batch.size(-1)
                )

                # Predict scores for the batch
                batch_scores = self.predict(
                    user_batch_expanded, item_sequences_batch, items_batch
                ).view(len(user_batch), -1)

                # Get top k items for each user in the batch
                topk_scores, topk_indices = torch.topk(batch_scores, k, dim=1)
                topk_items = all_items[topk_indices]

                # Collect recommendations
                user_indices.extend(user_batch.repeat_interleave(k).cpu().tolist())
                recommendations.extend(topk_items.cpu().flatten().tolist())
                scores.extend(topk_scores.cpu().flatten().tolist())

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