import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset


class UserItemRatingDFDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        user_col: str,
        item_col: str,
        rating_col: str,
        timestamp_col: str,
        item_seq_col: str = None
    ):
        self.df = df.assign(**{rating_col: df[rating_col].astype(np.float32)})
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.timestamp_col = timestamp_col
        self.item_seq_col = item_seq_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user = self.df[self.user_col].iloc[idx]
        item = self.df[self.item_col].iloc[idx]
        rating = self.df[self.rating_col].iloc[idx]
        item_sequence = self.df[self.item_seq_col].iloc[idx] if self.item_seq_col else None

        return dict(
            user=torch.as_tensor(user),
            item=torch.as_tensor(item),
            rating=torch.as_tensor(rating),
            item_sequence=torch.as_tensor(item_sequence) if item_sequence is not None else None,
        )

    @classmethod
    def get_default_loss_fn(cls):
        loss_fn = nn.MSELoss()
        return loss_fn

    @classmethod
    def forward(cls, model, batch_input, loss_fn=None, device="cpu"):
        predictions = model.predict_train_batch(batch_input, device=device).squeeze()
        ratings = batch_input["rating"].to(device).squeeze()

        if loss_fn is None:
            loss_fn = cls.get_default_loss_fn()
        loss = loss_fn(predictions, ratings)
        return loss
    
class SASRecDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        user_col: str = "user_id",
        seq_col: str = "sequence_item",
        item_col: str = "target",
        rating_col: str = "rating",
        maxlen: int = 10,
        pad_token: int = 0,
        timestamp_col: str = None,
    ):
        """
        Dataset for SASRec with left-padding:
        - Sequences are left-padded to reach maxlen
        - Binary target (0/1)

        Args:
            df: DataFrame containing the data
            user_col: Column name for user IDs
            seq_col: Column name for sequences (as lists)
            target_col: Column name for targets (0/1)
            maxlen: Maximum sequence length (after padding)
            pad_token: Padding token value
        """
        self.df = df
        self.user_col = user_col
        self.seq_col = seq_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.maxlen = maxlen
        self.pad_token = pad_token
        self.timestamp_col = timestamp_col
        self.num_items = df[item_col].max() if item_col in df else 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # raw data
        user = self.df[self.user_col].iloc[idx]
        seq = self.df[self.seq_col].iloc[idx]  # list
        item = self.df[self.item_col].iloc[idx]
        rating = self.df[self.rating_col].iloc[idx]
        seq = self._replace_padding(seq)

        return {
            "user": torch.as_tensor(user, dtype=torch.long),
            "sequence": torch.as_tensor(seq, dtype=torch.long),
            "item": torch.as_tensor(item, dtype=torch.long),
            "rating": torch.as_tensor(rating, dtype=torch.long),
        }
    def _replace_padding(self,seq):
        # replace the -1 elements with the num_items + 1  # +1 for padding token
        return np.where(seq == -1, self.num_items + 1, seq)

    def _left_pad(self, seq):
        if len(seq) > self.maxlen:
            return seq[-self.maxlen:]  
        elif len(seq) < self.maxlen:
            return [self.pad_token] * (self.maxlen - len(seq)) + seq  
        return seq

    @classmethod
    def get_default_loss_fn(cls):
        return nn.BCEWithLogitsLoss()

    @classmethod
    def forward(cls, model, batch_input, loss_fn=None, device="cpu"):
        users = batch_input["user"].to(device)
        seqs = batch_input["sequence"].to(device)
        targets = batch_input["target"].to(device)
        
        logits = model(users, seqs).squeeze()
        
        if loss_fn is None:
            loss_fn = cls.get_default_loss_fn()
        
        return loss_fn(logits, targets)