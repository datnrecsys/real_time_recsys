from typing import Any, Dict

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from src.algo.gSASRec.dataset import SASRecDataset

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout_rate)

        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)

    def forward(self, inputs):
        # outputs = self.dropout1(self.relu(self.conv1(inputs.transpose(-1, -2))))
        # outputs = self.dropout2(self.conv2(outputs).transpose(-1, -2))
        # return outputs

        x = inputs.transpose(-1, -2)         # [B, H, L]
        y = self.conv1(x)
        y = self.relu(y)
        z = self.dropout1(y)

        w = self.conv2(z)
        w = w.transpose(-1, -2)
        out = self.dropout2(w)
        return out


class SASRec(nn.Module):
    def __init__(self, user_num, item_num, hidden_units, dropout_rate, num_blocks, num_heads):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.seq_len = 10  # Fixed sequence length

        # Item and Position Embeddings
        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=item_num)
        self.pos_emb = nn.Embedding(self.seq_len, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)

        # Xavier initialization for embeddings
        nn.init.xavier_normal_(self.item_emb.weight)  
        self.item_emb.weight.data[item_num].zero_()   
        nn.init.xavier_normal_(self.pos_emb.weight)

        # Transformer Blocks
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        
        for _ in range(num_blocks):
            # Attention
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-5))
            self.attention_layers.append(nn.MultiheadAttention(
                embed_dim=hidden_units,
                num_heads=num_heads,
                dropout=dropout_rate,
                batch_first=True,
                add_zero_attn=True
            ))

            # Xavier initialization for MultiheadAttention
            for name, param in self.attention_layers.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)  # Initialize bias to 0
            
            # FFN
            self.forward_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-5))
            self.forward_layers.append(PointWiseFeedForward(hidden_units, dropout_rate))

        # Final layers
        self.final_layer = nn.Linear(hidden_units, 1)
        self.sigmoid = nn.Sigmoid()

        # Xavier initialization for final layer
        nn.init.xavier_uniform_(self.final_layer.weight)
        if self.final_layer.bias is not None:
            nn.init.zeros_(self.final_layer.bias)  # Initialize bias to 0

    def get_mask(self, seq):
        return (seq == self.item_num )

    @classmethod
    def get_default_dataset(cls):
        """
        Returns the expected dataset type for the model.

        Returns:
            UserItemRatingDFDataset: The expected dataset type.
        """
        return SASRecDataset

    def forward(self, user_ids, seq, target_item):
        assert seq.max() <= self.item_num, f"Invalid token index: {seq.max()} exceeds embedding size"
        
        seq_emb = self.item_emb(seq) * (self.hidden_units ** 0.5)

        if torch.isnan(seq_emb).any():
            print("\n=== NaN detected in seq_emb ===")
            print(seq_emb)
            print(seq_emb.shape)
            
            # Lấy thông tin batch có lỗi
            nan_mask = torch.isnan(seq_emb)
            batch_has_nan = nan_mask.any(dim=2).any(dim=1)
            problematic_indices = torch.where(batch_has_nan)[0]
            
            print(f"Batch size: {seq.size(0)}, Seq length: {seq.size(1)}")
            print(f"NaN detected in {len(problematic_indices)} sequences:")
            
            for idx in problematic_indices:
                print(f"\n--- Problematic Sequence {idx.item()} ---")
                print(f"User ID: {user_ids[idx].item()}")
                print(f"Target Item: {target_item[idx].item()}")
                print(f"Full Sequence: {seq[idx].tolist()}")
                
                # Tìm vị trí NaN cụ thể trong sequence
                seq_nan_mask = nan_mask[idx].any(dim=1)
                nan_positions = torch.where(seq_nan_mask)[0]
                print(f"NaN at positions: {nan_positions.tolist()}")
                
                # Kiểm tra các giá trị gây NaN
                problematic_items = seq[idx][seq_nan_mask]
                print(f"Problematic item indices: {problematic_items.tolist()}")
            
            return None
                
        # Add positional encoding
        positions = torch.arange(self.seq_len, device=seq.device).unsqueeze(0)
        seq_emb += self.pos_emb(positions)
        seq_emb = self.emb_dropout(seq_emb)
        
        mask = self.get_mask(seq)
        
        if mask.all(dim=1).any():
            print("Warning: Some sequences have all keys masked (entire sequence is padding).")
        
        assert mask.dtype == torch.bool, "key_padding_mask must be boolean"
        causal_mask = torch.triu(torch.ones(self.seq_len, self.seq_len), diagonal=1).bool().to(seq.device)
        assert causal_mask.dtype == torch.bool, "attn_mask must be boolean"
        
        assert seq_emb.shape[1] == mask.shape[1], "Sequence length mismatch between seq_emb and mask"
        
        # Transformer blocks
        for i in range(len(self.attention_layers)):
            # Self-attention
            seq_norm = self.attention_layernorms[i](seq_emb)
            attn_out, _ = self.attention_layers[i](
                query=seq_norm,
                key=seq_norm,
                value=seq_norm,
                attn_mask=causal_mask,
                key_padding_mask=mask
            )
            # lay thong tin batch co loi
            if torch.isnan(attn_out).any():
                print("\n=== NaN detected in attn_out ===")
                print(attn_out)
                print(attn_out.shape)
                
                # Lấy thông tin batch có lỗi
                nan_mask = torch.isnan(attn_out)
                batch_has_nan = nan_mask.any(dim=2).any(dim=1)
                problematic_indices = torch.where(batch_has_nan)[0]
                
                print(f"Batch size: {seq.size(0)}, Seq length: {seq.size(1)}")
                print(f"NaN detected in {len(problematic_indices)} sequences:")
                
                for idx in problematic_indices:
                    print(f"\n--- Problematic Sequence {idx.item()} ---")
                    print(f"User ID: {user_ids[idx].item()}")
                    print(f"Target Item: {target_item[idx].item()}")
                    print(f"Full Sequence: {seq[idx].tolist()}")
                    
                    # Tìm vị trí NaN cụ thể trong sequence
                    seq_nan_mask = nan_mask[idx].any(dim=1)
                    nan_positions = torch.where(seq_nan_mask)[0]
                    print(f"NaN at positions: {nan_positions.tolist()}")
                    
            seq_emb = seq_emb + attn_out
            # FFN
            seq_emb = self.forward_layers[i](self.forward_layernorms[i](seq_emb)) + seq_emb
        # Get final state
        final_state = seq_emb[:, -1, :]  # Last item in sequence

        # Predict target item score
        target_emb = self.item_emb(target_item)
        logits = (final_state * target_emb).sum(dim=-1)
        return logits

    def predict(self, user_ids, seq, target_item):
        return self.sigmoid(self.forward(user_ids, seq, target_item))

    def recommend(self, users, seqs, k=10):
        self.eval()
        all_items = torch.arange(0, self.item_num, device=seqs.device)
        scores = []
        with torch.no_grad():
            user_len_debug = 10
            
            for i in range(len(users)):
                # print(i)
                if i == user_len_debug:
                    break
                seq = seqs[i].unsqueeze(0).repeat(self.item_num, 1)                
                items = all_items#.unsqueeze(1)
                user = users[i].repeat(self.item_num, 1).squeeze(1)
                score = self.predict(user, seq, items)
                scores.append(score)
        topk = torch.stack(scores).topk(k)
        
        return {
            'user_indice': users[0:user_len_debug].cpu().numpy().tolist(), 
            'recommendation': topk.indices[0:user_len_debug].cpu().numpy().tolist(), 
            'score': topk.values[0:user_len_debug].cpu().numpy().tolist()  
        }
