import torch
from torch import nn
import numpy as np

from src.algo.gSASRec.transformer_decoder import MultiHeadAttention

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, query_states, key_states=None, seq_len=None, position_ids=None):
        seq_len = query_states.shape[2] if not seq_len else seq_len
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=query_states.device, dtype=query_states.dtype)
        cos = self.cos_cached[:, :, :seq_len, ...].to(dtype=query_states.dtype)
        sin = self.sin_cached[:, :, :seq_len, ...].to(dtype=query_states.dtype)
        position_ids = torch.arange(seq_len, device=query_states.device).unsqueeze(0) if position_ids is None else position_ids
        query_states = apply_rotary_pos_emb(query_states, cos, sin, position_ids)
        if key_states is not None:
            key_states = apply_rotary_pos_emb(key_states, cos, sin, position_ids)
            return query_states, key_states
        return query_states

class CAPE(nn.Module):
    def __init__(self, dim, max_len=None, embedding_dim=None):
        super().__init__()
        self.max_len = max_len if max_len is not None else dim
        self.pos_emb = nn.parameter.Parameter(torch.zeros(1, dim, self.max_len))
        if embedding_dim is not None:
            self.pre_proj = nn.Sequential(nn.Linear(embedding_dim, dim), nn.SiLU())

    def forward(self, query, attention_weights):
        G = 1 - torch.sigmoid(attention_weights)
        P = G.flip(-1).cumsum(dim=-1).flip(-1)
        P = P.clamp(max=self.max_len - 1)
        P_ceil = P.ceil().long()
        P_floor = P.floor().long()
        if getattr(self, 'pre_proj', None) is not None:
            query = self.pre_proj(query)
        E = torch.matmul(query, self.pos_emb)
        E_ceil = E.gather(-1, P_ceil)
        E_floor = E.gather(-1, P_floor)
        P_P_floor = P - P_floor
        E = P_P_floor * E_ceil + (1 - P_P_floor) * E_floor
        return E

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs
    
class SASRec_CAPE(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec_CAPE, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.args = args

        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        # No need for position embeddings anymore as we're using CAPE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.cape = CAPE(dim=args.hidden_units, max_len=args.maxlen, embedding_dim=args.hidden_units)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            self.attention_layernorms.append(torch.nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.attention_layers.append(
                torch.nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate)
            )
            self.forward_layernorms.append(torch.nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(args.hidden_units, args.dropout_rate))

    def log2feats(self, log_seqs):
        seqs = self.item_emb(log_seqs)  # (batch_size, seq_len, hidden_units)
        seqs *= self.item_emb.embedding_dim ** 0.5

        # No positional embedding here, using CAPE instead after attention calculation
        seqs = self.emb_dropout(seqs)
        tl = seqs.size(1)
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)  # (seq_len, batch_size, hidden_units)
            Q = self.attention_layernorms[i](seqs)

            # Get attention weights from MultiheadAttention
            mha_outputs, attn_output_weights = self.attention_layers[i](
                Q, seqs, seqs, attn_mask=attention_mask, need_weights=True
            )
            seqs = Q + mha_outputs  # Residual connection

            # Integrate CAPE
            batch_size, seq_len, hidden_units = seqs.shape[1], seqs.shape[0], seqs.shape[2]
            num_heads = self.args.num_heads
            head_dim = hidden_units // num_heads

            # Reshape Q to (batch_size * num_heads, seq_len, head_dim)
            Q_reshaped = Q.view(seq_len, batch_size, num_heads, head_dim).transpose(0, 1) \
                            .reshape(batch_size * num_heads, seq_len, head_dim)

            # Reshape attn_output_weights to (batch_size * num_heads, seq_len, seq_len)
            attn_weights = attn_output_weights.view(batch_size * num_heads, seq_len, seq_len)

            # Call CAPE
            cape_output = self.cape(Q_reshaped, attn_weights)  # (batch_size * num_heads, seq_len, seq_len)

            # Reshape cape_output back to (batch_size, seq_len, hidden_units)
            cape_emb = cape_output.mean(dim=-1)  # (batch_size * num_heads, seq_len)
            cape_emb = cape_emb.view(batch_size, num_heads, seq_len, 1).transpose(1, 2) \
                                .repeat(1, 1, 1, head_dim).view(batch_size, seq_len, hidden_units)

            # Add cape_emb to seqs
            seqs = torch.transpose(seqs, 0, 1)  # (batch_size, seq_len, hidden_units)
            seqs += cape_emb

            # Continue with forward layers
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)
        
        pos_embs = self.item_emb(pos_seqs)  # pos_seqs is already a tensor on GPU
        neg_embs = self.item_emb(neg_seqs)  # neg_seqs is already a tensor on GPU
        
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        
        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(item_indices.to(self.dev))
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits
        
class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            # new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
            #                                                 args.num_heads,
            #                                                 args.dropout_rate)

            new_attn_layer = MultiHeadAttention(args.hidden_units, 
                                                args.num_heads, 
                                                args.dropout_rate)
            
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs):
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        
        # Create positions using PyTorch
        positions = torch.arange(log_seqs.size(1), device=self.dev).unsqueeze(0) + 1  # Start from 1
        positions = positions.expand(log_seqs.size(0), -1)  # Tile by batch size
        
        # Clone tensor before performing in-place operation
        positions = positions.clone()  # <-- Add this line to avoid memory overlap
        pos_mask = (log_seqs != 0).long()
        positions *= pos_mask  # Now safe to perform
        
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)
        
        # Rest stays unchanged
        tl = seqs.size(1)
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)
            
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
        
        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        # log_seqs: [B, L], pos_seqs: [B, L], neg_seqs: [B, L, N]
        # log_feats: [B, L, H], pos_embs: [B, L, H], neg_embs: [B, L, N, H]
        log_feats = self.log2feats(log_seqs) 
        
        pos_embs = self.item_emb(pos_seqs)  
        neg_embs = self.item_emb(neg_seqs)  

        # pos_logits = (log_feats * pos_embs).sum(dim=-1)
        # neg_logits = (log_feats * neg_embs).sum(dim=-1)
        pos_logits = (log_feats * pos_embs).sum(dim=-1)  
        neg_logits = (log_feats.unsqueeze(2) * neg_embs).sum(dim=-1)  
        
        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(item_indices.to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)
