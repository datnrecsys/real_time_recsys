import importlib
import matplotlib.pyplot as plt

from src.algo.gSASRec.config import gSASRecExperimentConfig

def load_config(config_file: str) -> gSASRecExperimentConfig:
    spec = importlib.util.spec_from_file_location("config", config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config

def print_first_batch(dataloader, shuffle_status):
    for uid, seq, pos, neg in dataloader:
        print("First batch with shuffle status:", shuffle_status)
        print("uid shape:", uid.shape, "uid value:", uid[:1])
        print("seq shape:", seq.shape, "seq value:", seq[:1])
        print("pos shape:", pos.shape, "pos value:", pos[:1])
        print("neg shape:", neg.shape, "neg value:", neg[:1])
        break

def plot_combined_metrics(metrics_df, config):    
    plt.figure(figsize=(12, 6))
    
    ax1 = plt.gca()
    
    ax1.plot(metrics_df['epoch'], metrics_df['train_loss'], 'b-', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax2 = ax1.twinx()
    
    eval_metrics = metrics_df.dropna(subset=['val_hr'])
    epochs = eval_metrics['epoch']
    
    width = 3  
    ax2.bar(epochs - width/2, eval_metrics['val_hr'], width, label='Val HR@10', color='orange', alpha=0.6)
    ax2.bar(epochs + width/2, eval_metrics['val_ndcg'], width, label='Val NDCG@10', color='green', alpha=0.6)
    
    ax2.set_ylabel('Metrics Score', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title('Training Loss and Evaluation Metrics')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{config.train_dir}/training_metrics.png', dpi=300)
    # plt.show()