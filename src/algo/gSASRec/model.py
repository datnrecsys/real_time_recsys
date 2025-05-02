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

        # # hook cho conv1
        # self.conv1.register_full_backward_hook(self._make_conv_hook("conv1"))
        # # hook cho conv2
        # self.conv2.register_full_backward_hook(self._make_conv_hook("conv2"))
        
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity="relu")
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)
        # conv2 is linear, so Xavier is fine there
        nn.init.xavier_uniform_(self.conv2.weight, gain=0.1)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

    # def _make_conv_hook(self, name):
    #     def hook(module, grad_input, grad_output):
    #         # grad_input, grad_output là tuple của các tensor gradients
    #         for i, g in enumerate(grad_input):
    #             if g is not None and torch.isnan(g).any():
    #                 print(f"NaN in {name}.grad_input[{i}]")
    #         for i, g in enumerate(grad_output):
    #             if g is not None and torch.isnan(g).any():
    #                 print(f"NaN in {name}.grad_output[{i}]")
    #         return None  # không modify gradient
    #     return hook

    def forward(self, inputs):
        # outputs = self.dropout1(self.relu(self.conv1(inputs.transpose(-1, -2))))
        # outputs = self.dropout2(self.conv2(outputs).transpose(-1, -2))
        # return outputs

        x = inputs.transpose(-1, -2)         # [B, H, L]
        y = self.conv1(x)
        # tensor-level hook: in ra ngay khi grad chảy qua y
        # y.register_hook(lambda grad: print("grad of conv1 output has NaN") 
        #                 if torch.isnan(grad).any() else None)
        y = self.relu(y)
        z = self.dropout1(y)
        # z = torch.clamp(z, -1e3, 1e3)  # clamp để tránh NaN trong dropout
        # cũng hook lên z trước conv2
        # z.register_hook(lambda grad: print("grad of conv1 - dropout1 output has NaN") 
        #                 if torch.isnan(grad).any() else None)

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
        nn.init.xavier_uniform_(self.item_emb.weight)  
        self.item_emb.weight.data[item_num].zero_()   
        nn.init.xavier_uniform_(self.pos_emb.weight)

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
                    nn.init.xavier_uniform_(param)
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
        
        seq_emb = self.item_emb(seq) #* (self.hidden_units ** 0.5)
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

        # if torch.isnan(seq_emb).any():
        #     print("NaN values found in seq_emb")
        
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
                    
                    # Kiểm tra các giá trị
            
            seq_emb = seq_emb + attn_out
            # print(f"seq_emb", seq_emb)
            # FFN
            seq_emb = self.forward_layers[i](self.forward_layernorms[i](seq_emb)) + seq_emb
            # print(f"seq_emb", seq_emb)
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
            # print(seqs.shape)
            # print(all_items.shape)
            # print(users.shape)
            user_len_debug = 10
            
            for i in range(len(users)):
                # print(i)
                if i == user_len_debug:
                    break
                seq = seqs[i].unsqueeze(0).repeat(self.item_num, 1)                
                items = all_items#.unsqueeze(1)
                user = users[i].repeat(self.item_num, 1).squeeze(1)
                # print("seq shape",seq.shape)
                # print(seq)
                # print("items shape",items.shape)
                # print("user shape",user.shape)
                score = self.predict(user, seq, items)
                # print("score shape",score.shape)
                scores.append(score)
        # print(len(scores))
        # print(scores[0].shape)
        # print(scores[0])
        topk = torch.stack(scores).topk(k)
        # print(topk.values.shape)
        # print("DEBUG")
        
        return {
            'user_indice': users[0:user_len_debug].cpu().numpy().tolist(), 
            'recommendation': topk.indices[0:user_len_debug].cpu().numpy().tolist(), 
            'score': topk.values[0:user_len_debug].cpu().numpy().tolist()  
        }
