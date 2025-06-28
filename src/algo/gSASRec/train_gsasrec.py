import os
import time

import numpy as np
import pandas as pd
import torch

from src.algo.gSASRec.config_electronics import config
from src.algo.gSASRec.model import SASRec, SASRec_CAPE
from src.algo.gSASRec.pre_process_data import pre_process_data
from src.algo.gSASRec.utils import (load_config, plot_combined_metrics,
                                    print_first_batch)
from src.algo.gSASRec.utils_dataset import *
from src.algo.gSASRec.utils_validation import evaluate, evaluate_valid

# Load the configuration file
# run in kaggle kernel
# config = load_config('/kaggle/usr/lib/config-electronics/config_electronics.py')
# df_train = pd.read_parquet('/kaggle/input/amazon-electronics-0-1/train_sample_interactions_16407u.parquet')
# df_val = pd.read_parquet('/kaggle/input/amazon-electronics-0-1/test_sample_interactions_16407u.parquet')
# df_test = pd.read_parquet('/kaggle/input/amazon-electronics-0-1/val_sample_interactions_16407u.parquet')

if not os.path.exists(config.dataset_dir):
    os.makedirs(config.dataset_dir)

# run in local machine
config = load_config('config_electronics.py')
df_train = pd.read_parquet(f'{config.dataset_dir}/train_sample_interactions_16407u.parquet')
df_val = pd.read_parquet(f'{config.dataset_dir}/test_sample_interactions_16407u.parquet')
df_test = pd.read_parquet(f'{config.dataset_dir}/val_sample_interactions_16407u.parquet')

# Load the dataset
data_full = pd.DataFrame()
data_full = pd.concat([data_full, df_train, df_val], ignore_index=True)

# if file Electronics.txt exists, delete it
if os.path.exists(f"{config.dataset_dir}/{config.categories[0]}.txt"):
    os.remove(f"{config.dataset_dir}/{config.categories[0]}.txt")
usermap, usernum, itemmap, itemnum = pre_process_data(data_full, config)

group_history_transaction(data_full,config.output_normal_sequence)
convert_sequences_to_ids(config.output_normal_sequence,itemmap)

group_history_transaction(df_test,f"{config.dataset_dir}/sequences-test.txt")
convert_sequences_to_ids(f"{config.dataset_dir}/sequences-test.txt",itemmap)

if config.sampling_method == "graph" or config.sampling_method == "hybrid":
    generator = GraphSequenceGenerator(
        data=data_full,
        output_path = config.output_graph_sequence,
        sequence_len = config.sequence_len,
        num_sequence = config.num_sequence,
        least_interaction = config.least_interaction,
        name= config.name,
    )
    generator.generate_sequences(isTrain=True)
    sequences = generator.sample_array
    convert_sequences_to_ids(config.output_graph_sequence,itemmap)

if not os.path.isdir(config.train_dir):
    os.makedirs(config.train_dir)
with open(os.path.join(config.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([f"{k},{v}" for k, v in sorted(vars(config).items(), key=lambda x: x[0])]))
f.close()

dataset_test = data_partition(f"{config.dataset_dir}/sequences-test_ids.txt")
if config.sampling_method == 'graph':
    dataset = data_partition(f"{config.dataset_dir}/sequences-graph_ids.txt")
    
elif config.sampling_method == 'normal':
    dataset = data_partition(f"{config.dataset_dir}/sequences-normal_ids.txt")
    
elif config.sampling_method == 'hybrid':
    normal_sample_dir = f"{config.dataset_dir}/sequences-normal_ids.txt"
    graph_sample_dir = f"{config.dataset_dir}/sequences-graph_ids.txt"
    fused_set = fuse_data(normal_sample_dir, normal_sample_dir, 0.3,f"{config.dataset_dir}/sequence-hybrid_ids.txt")
    dataset = data_partition(f"{config.dataset_dir}/sequence-hybrid_ids.txt")

[user_train, user_valid, user_test, usernum, itemnum] = dataset
num_batch = (len(user_train) - 1) // config.batch_size + 1
cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print(f'average sequence length: {cc / len(user_train):.2f}')

# create log file
f = open(os.path.join(config.train_dir, 'log.txt'), 'w')
f.write('epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n')
f.write('------------------------------------------------\n')
f.flush()

# initialize the model and sampler
sampler = get_dataloader(user_train=user_train, usernum=usernum, 
                         itemnum=itemnum, maxlen=config.maxlen, 
                         batch_size=config.batch_size, args=config, )
print_first_batch(sampler, shuffle_status=True)
    
if config.pe_method == "cape":
    print("loading cape + sasrec model")
    model = SASRec_CAPE(usernum, itemnum, config).to(config.device)
elif config.pe_method == "normal":
    print("loading sasrec model")
    model = SASRec(usernum, itemnum, config).to(config.device)

# initialize the model parameters
for name, param in model.named_parameters():
    try:
        torch.nn.init.xavier_normal_(param.data)
    except:
        pass  # skip if the parameter is not a tensor

if config.pe_method == "normal":
    model.pos_emb.weight.data[0, :] = 0
model.item_emb.weight.data[0, :] = 0

model.train() 
# load pretrained model if exists
epoch_start_idx = 1
if config.state_dict_path is not None:
    try:
        model.load_state_dict(torch.load(config.state_dict_path, 
                              map_location=torch.device(config.device)))
        tail = config.state_dict_path[config.state_dict_path.find('epoch=') + 6:]
        epoch_start_idx = int(tail[:tail.find('.')]) + 1
    except:
        print('failed loading state_dicts, pls check file path: ', end="")
        print(config.state_dict_path)

# initialize the optimizer and loss function
bce_criterion = torch.nn.BCEWithLogitsLoss()
adam_optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.98))

best_val_ndcg, best_val_hr = 0.0, 0.0
best_test_ndcg, best_test_hr = 0.0, 0.0
T = 0.0
train_losses = []
val_metrics = {'epoch': [], 'HR@10': [], 'NDCG@10': []}
test_metrics = {'epoch': [], 'HR@10': [], 'NDCG@10': []}
t0 = time.time()

for epoch in range(epoch_start_idx, config.num_epochs + 1):

    epoch_loss = 0
    num_batches = 0

    for step, (u, seq, pos, neg) in enumerate(sampler):
        
        u = torch.tensor(u, dtype=torch.long, device=config.device)
        seq = torch.tensor(np.array(seq), dtype=torch.long, device=config.device)
        pos = torch.tensor(np.array(pos), dtype=torch.long, device=config.device)
        neg = torch.tensor(np.array(neg), dtype=torch.long, device=config.device)
        
        pos_logits, neg_logits = model(u, seq, pos, neg)

        pos_labels = torch.ones(pos_logits.shape, device=config.device)
        neg_labels = torch.zeros(neg_logits.shape, device=config.device)
        
        adam_optimizer.zero_grad()
        indices = np.where(pos.cpu() != 0)

        alpha = config.num_negs / (itemnum - 1)
        t = config.gbce_t 
        beta = alpha * ((1 - 1/alpha)*t + 1/alpha)
        positive_logits = pos_logits.to(torch.float64)
        negative_logits = neg_logits.to(torch.float64)
        eps = 1e-10
        positive_probs = torch.clamp(torch.sigmoid(positive_logits), eps, 1-eps)
        positive_probs_adjusted = torch.clamp(positive_probs.pow(-beta), 1+eps, torch.finfo(torch.float64).max)
        to_log = torch.clamp(torch.div(1.0, (positive_probs_adjusted  - 1)), eps, torch.finfo(torch.float64).max)
        positive_logits_transformed = to_log.log()

        # loss = bce_criterion(pos_logits[indices], pos_labels[indices])
        # loss += bce_criterion(neg_logits[indices], neg_labels[indices])
        pos_loss = bce_criterion(positive_logits_transformed[indices], pos_labels[indices])
        neg_loss = bce_criterion(neg_logits[indices], neg_labels[indices]).mean(dim=-1) 

        loss = pos_loss + neg_loss
        
        for param in model.item_emb.parameters():
            loss += config.l2_emb * torch.norm(param)
        
        epoch_loss += loss.item()
        num_batches += 1

        loss.backward()
        adam_optimizer.step()

    print(f"loss in epoch {epoch} iteration {step}: {loss.item()}")
    avg_loss = epoch_loss / num_batches
    train_losses.append(avg_loss)

    if epoch % 10 == 0:
        model.eval()
        t1 = time.time() - t0
        T += t1
        print('Evaluating', end='')
        t_test = evaluate(model, dataset, config)
        t_valid = evaluate_valid(model, dataset, config)

        val_metrics['epoch'].append(epoch)
        val_metrics['HR@10'].append(t_valid[1])
        val_metrics['NDCG@10'].append(t_valid[0])
        
        test_metrics['epoch'].append(epoch)
        test_metrics['HR@10'].append(t_test[1])
        test_metrics['NDCG@10'].append(t_test[0])

        print(f'epoch:{epoch}, time: {T:f}(s), valid (NDCG@10: {t_valid[0]:.4f}, HR@10: {t_valid[1]:.4f}), test (NDCG@10: {t_test[0]:.4f}, HR@10: {t_test[1]:.4f})')

        if t_valid[0] > best_val_ndcg or t_valid[1] > best_val_hr or t_test[0] > best_test_ndcg or t_test[1] > best_test_hr:
            best_val_ndcg = max(t_valid[0], best_val_ndcg)
            best_val_hr = max(t_valid[1], best_val_hr)
            best_test_ndcg = max(t_test[0], best_test_ndcg)
            best_test_hr = max(t_test[1], best_test_hr)
            folder = config.train_dir
            fname = f'SASRec.epoch={epoch}.lr={config.lr}.layer={config.num_blocks}.head={config.num_heads}.hidden={config.hidden_units}.maxlen={config.maxlen}.pth'
            torch.save(model.state_dict(), os.path.join(folder, fname))

        f.write(f'{epoch} {t_valid} {t_test}\n')
        f.flush()
        t0 = time.time()
        model.train()
    
    if epoch == config.num_epochs:
        folder = config.train_dir
        fname = f'SASRec.epoch={config.num_epochs}.lr={config.lr}.layer={config.num_blocks}.head={config.num_heads}.hidden={config.hidden_units}.maxlen={config.maxlen}.pth'
        torch.save(model.state_dict(), os.path.join(folder, fname))

f.close()

metrics_df = pd.DataFrame({
    'epoch': list(range(1, config.num_epochs + 1)),
    'train_loss': train_losses,
    'val_hr': [None] * len(train_losses),
    'val_ndcg': [None] * len(train_losses),
    'test_hr': [None] * len(train_losses),
    'test_ndcg': [None] * len(train_losses)
})

for i, epoch in enumerate(val_metrics['epoch']):
    idx = epoch - 1
    metrics_df.at[idx, 'val_hr'] = val_metrics['HR@10'][i]
    metrics_df.at[idx, 'val_ndcg'] = val_metrics['NDCG@10'][i]
    metrics_df.at[idx, 'test_hr'] = test_metrics['HR@10'][i]
    metrics_df.at[idx, 'test_ndcg'] = test_metrics['NDCG@10'][i]

metrics_df.to_csv(f'{config.train_dir}/training_metrics.csv', index=False)
plot_combined_metrics(metrics_df, config)
print("Done")