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

    def forward(self, inputs):
        outputs = self.dropout1(self.relu(self.conv1(inputs.transpose(-1, -2))))
        outputs = self.dropout2(self.conv2(outputs).transpose(-1, -2))
        return outputs

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

        # Transformer Blocks
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        
        for _ in range(num_blocks):
            # Attention
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.attention_layers.append(nn.MultiheadAttention(
                embed_dim=hidden_units,
                num_heads=num_heads,
                dropout=dropout_rate,
                batch_first=True
            ))
            
            # FFN
            self.forward_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(hidden_units, dropout_rate))

        # Final layers
        self.final_layer = nn.Linear(hidden_units, 1)
        self.sigmoid = nn.Sigmoid()

    def get_mask(self, seq):
        return (seq == self.item_num + 1)

    @classmethod
    def get_default_dataset(cls):
        """
        Returns the expected dataset type for the model.

        Returns:
            UserItemRatingDFDataset: The expected dataset type.
        """
        return SASRecDataset

    def forward(self, user_ids, seq, target_item):
        # Embedding
        # print(self.seq_len)
        # print("seq.shape", seq.shape)
        # print(seq)
        # print("target_item.shape", target_item.shape)
        # print(target_item)
        # print("user_ids.shape", user_ids.shape)
        # print(user_ids)
        seq_emb = self.item_emb(seq) * (self.hidden_units ** 0.5)
        positions = torch.arange(self.seq_len, device=seq.device).unsqueeze(0)
        seq_emb += self.pos_emb(positions)
        seq_emb = self.emb_dropout(seq_emb)

        # Transformer
        mask = self.get_mask(seq)
        causal_mask = torch.triu(torch.ones(self.seq_len, self.seq_len), diagonal=1).bool().to(seq.device)
        
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
        # print("DEBUG")
        # print(seq.shape)
        # print(target_item.shape)
        # print(user_ids.shape)
        
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
                # if i == user_len_debug:
                #     break
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
            'user_indice': users.cpu().numpy(),
            'recommendation': topk.indices.cpu().numpy(),
            'score': topk.values.cpu().numpy()
        }

# Test case
# if __name__ == "__main__":
#     class Args:
#         hidden_units = 64
#         dropout_rate = 0.1
#         num_blocks = 2
#         num_heads = 2

#     args = Args()
#     model = SASRec(
#         user_num=100,  # Not used in current architecture
#         item_num=50,
#         args=args
#     )

#     # Sample input
#     user_ids = torch.LongTensor([1, 2, 3])
#     seq = torch.LongTensor([
#         [1,2,3,4,5,0,0,0,0,0],
#         [6,7,8,9,10,11,0,0,0,0],
#         [12,13,0,0,0,0,0,0,0,0]
#     ])
#     target = torch.LongTensor([5, 12, 15])

#     # Test forward
#     logits = model(user_ids, seq, target)
#     print("Logits shape:", logits.shape)  # Should be [3]

#     # Test recommend
#     rec = model.recommend(user_ids[:1], seq[:1])
#     print("Recommendations:", rec)
    


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
