import sys
import copy
import torch
import os
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
import csv
import itertools
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader


def build_index(dataset_name):

    ui_mat = np.loadtxt('data/%s.txt' % dataset_name, dtype=np.int32)

    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def group_history_transaction(data, output_path, isTrain=True):
    """Process the raw data, generate the sequences of transactions for each user and its statistics."""
    # base_path = os.path.splitext(output_path)[0]
    # suffix = "" if isTrain else "-test"
    # output_file = f"{base_path}-input{suffix}.txt"
    
    print("Processing transaction history...")
    grouped = data.groupby('user_id')
    
    sequence_lengths = []
    sample_sequences = []
    
    with open(output_path, 'w') as f:
        for user_id, group in grouped:
            items = [str(i) for i in group['parent_asin'].tolist()]
            sequence_length = len(items)
            
            sequence_lengths.append(sequence_length)
            if len(sample_sequences) < 3:  # take only 3 samples
                sample_sequences.append(items)
            
            f.write(' '.join(items) + '\n')
    
    # calculate statistics
    max_len = max(sequence_lengths)
    min_len = min(sequence_lengths)
    avg_len = sum(sequence_lengths) / len(sequence_lengths)
    
    print("\n=== SEQUENCE STATISTICS ===")
    print(f"Total sequences: {len(sequence_lengths)}")
    print(f"Max sequence length: {max_len}")
    print(f"Min sequence length: {min_len}")
    print(f"Average sequence length: {avg_len:.2f}")
    print("\nSample sequences:")
    for i, seq in enumerate(sample_sequences[:3]):  
        print(f"Sample {i+1} (length {len(seq)}): {' -> '.join(seq[:])}{'...' if len(seq) > 5 else ''}")
    
    print(f"\nSaved transaction history to {output_path}")
    return None



def fuse_data(normal_path: str, graph_path: str, graph_ratio: float,output_path):
    """
    Combines data from two sampling methods
    Args:
        normal_path: Path to samples file from normal method
        graph_path: Path to samples file from graph method
        graph_ratio: Ratio of samples from graph (0.0-1.0)
    Returns:
        List of combined sequences
    """
    # Read data from both files
    with open(normal_path, 'r') as f:
        normal_sequences = [line.strip() for line in f]
    print(f"Number of sequences in {normal_path}: {len(normal_sequences)}")
    print(f"Example of first sequence from {normal_path}: {normal_sequences[:1]}")
    
    with open(graph_path, 'r') as f:
        graph_sequences = [line.strip() for line in f]
    print(f"Number of sequences in {graph_path}: {len(graph_sequences)}")
    print(f"Example of first sequence from {graph_path}: {graph_sequences[:1]}")
    
    # Take all samples from normal
    fused = normal_sequences.copy()
    
    # Take a percentage from graph
    num_graph_samples = int(len(graph_sequences) * graph_ratio)
    print(f"Number of sequences taken from {graph_path} (ratio {graph_ratio}): {num_graph_samples}")
    fused += random.sample(graph_sequences, num_graph_samples)
    
    # Shuffle to avoid order bias
    random.shuffle(fused)
    print(f"Total number of sequences in result: {len(fused)}")
    print(f"Example of first 3 sequences in result: {fused[:1]}")
    
    # Write results to .txt file
    with open(output_path, 'w') as f:
        for sequence in fused:
            f.write(sequence + '\n')
    
    return None
    
def data_partition(fname):
    """
    Reads file containing sequences created from graph-based sampling.
    Each line contains a sequence of items separated by spaces.
    For each sequence:
      - If sequence length < 3, skip (not enough for train/valid/test)
      - Train: entire sequence except the last 2 elements
      - Valid: second to last element
      - Test: last element
    Returns [user_train, user_valid, user_test, usernum, itemnum].
    Note: here, you can assign a virtual user_id to each sequence.
    """
    user_train = {}
    user_valid = {}
    user_test = {}
    usernum = 0
    item_set = set()
    
    with open(fname, 'r') as f:
        for line in f:
            # Assume each line is a list of item IDs separated by spaces
            items = [int(x) for x in line.strip().split()]
            if len(items) < 3:
                continue  # not enough for train/valid/test
            usernum += 1
            uid = usernum  # assign virtual user_id
            user_train[uid] = items[:-2]
            user_valid[uid] = [items[-2]]
            user_test[uid] = [items[-1]]
            item_set.update(items)
    
    itemnum = max(item_set) if item_set else 0
    return [user_train, user_valid, user_test, usernum, itemnum]

class SASRecDataset(Dataset):
    def __init__(self, user_train, usernum, itemnum, maxlen, num_negs=1):
        self.user_train = user_train
        self.usernum = usernum
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.num_negs = num_negs  # Số lượng negative samples cho mỗi positive
        self.users = [uid for uid in range(1, usernum+1) if len(user_train[uid]) > 1]
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, index):
        uid = self.users[index]
        seq = np.zeros(self.maxlen, dtype=np.int32)
        pos = np.zeros(self.maxlen, dtype=np.int32)
        neg = np.zeros((self.maxlen, self.num_negs), dtype=np.int32)  
        
        seq_ = self.user_train[uid][:-1]
        pos_ = self.user_train[uid][1:]
        rated = set(self.user_train[uid])
        
        # Padding from the left
        idx = self.maxlen - 1
        for i in reversed(range(len(seq_))):
            seq[idx] = seq_[i]
            pos[idx] = pos_[i]
            # generate negative samples
            for k in range(self.num_negs):
                neg[idx, k] = self._random_neg(rated)
            idx -= 1
            if idx == -1:
                break
                
        return uid, seq, pos, neg
    
    def _random_neg(self, rated):
        t = np.random.randint(1, self.itemnum + 1)
        while t in rated:
            t = np.random.randint(1, self.itemnum + 1)
        return t
        
def get_dataloader(user_train, usernum, itemnum, maxlen, batch_size, args):
    dataset = SASRecDataset(user_train, usernum, itemnum, maxlen, args.num_negs)
    dataloader = DataLoader(dataset, num_workers=args.num_workers, 
                            batch_size=batch_size, shuffle=True,)
    return dataloader

def convert_sequences_to_ids(output_path,product_to_id_dict: dict):
    """
    Convert product sequences to IDs based on mapping dictionary
    
    Args:
        product_to_id_dict: Dictionary mapping product to ID (e.g. {'B07J3GH1W1': 1})
    
    Returns:
        List of sequences converted to IDs
    """
    # Read sequences.txt file
    sequences = []
    with open(output_path, 'r') as f:
        for line in f:
            products = line.strip().split()
            # Convert each product to ID
            sequence_ids = [str(product_to_id_dict.get(product, 0)) for product in products]
            sequences.append(sequence_ids)
    
    # Create path for new output file
    base_path = os.path.splitext(output_path)[0]
    id_output_path = f"{base_path}_ids.txt"
    
    # Save ID sequences to new file
    with open(id_output_path, 'w') as f:
        for seq in sequences:
            f.write(' '.join(seq) + '\n')
    
    print(f"Saved ID sequences to {id_output_path}")
    return None

class GraphSequenceGenerator:
    def __init__(self, data: pd.DataFrame, output_path: str, sequence_len: int = 40, num_sequence: int = 10, 
                 least_interaction: int = 1, name: str = 'Amazon'):
        """
        Initialize generator
        Args:
            data: DataFrame containing original data with columns: user_id, parent_asin
            output_path: Output file path (e.g. 'data/sequences.txt')
            sequence_len: Length of each sequence
            num_sequence: Number of sequences to generate for each node
            least_interaction: Minimum interaction frequency threshold
            name: Dataset name
        """
        self.data = data
        self.output_path = output_path
        self.sequence_len = sequence_len
        self.num_sequence = num_sequence
        self.least_interaction = least_interaction
        self.name = name
                
        # Intermediate variables
        self.frequency_dict = defaultdict(int)
        self.graph = None
        self.node_dict = {}
        self.transition_matrix = None
        self.transition_dict = {}
        self.sample_array = None

    def _group_history_transaction(self, isTrain=True):
        """Process original data, write to item sequences file"""
        base_path = os.path.splitext(self.output_path)[0]  # Remove extension
        suffix = "" if isTrain else "-test"
        output_file = f"{base_path}-input{suffix}.txt"
        
        print("Processing transaction history...")
        grouped = self.data.groupby('user_id')
        
        with open(output_file, 'w') as f:
            for user_id, group in grouped:
                items = [str(i) for i in group['parent_asin'].tolist()]
                f.write(' '.join(items) + '\n')
        print(f"Saved transaction history to {output_file}")
        return output_file

    def _get_frequency_pairs(self, input_file: str, isTrain=True):
        """Create edgelist from item pairs"""
        base_path = os.path.splitext(self.output_path)[0]
        output_file = f"{base_path}.edgelist"
        
        self.frequency_dict = defaultdict(int)
        
        with open(input_file, 'r') as f:
            for line in f:
                items = line.strip().split()
                for pair in itertools.combinations(items, 2):
                    sorted_pair = tuple(sorted(pair))
                    self.frequency_dict[sorted_pair] += 1
        
        with open(output_file, 'w') as f:
            for (i1, i2), freq in self.frequency_dict.items():
                if freq >= self.least_interaction:
                    f.write(f"{i1} {i2} {freq}\n")
        
        print(f"Saved frequency pairs to {output_file}")
        return output_file

    def _build_graph(self, edgelist_path: str):
        """Build graph from edgelist"""
        print("Building graph...")
        self.graph = nx.read_weighted_edgelist(edgelist_path)
        self.node_dict = {i: n for i, n in enumerate(self.graph.nodes())}
        print(f"Graph built with {len(self.graph.nodes())} nodes, {len(self.graph.edges())} edges")

    def _create_transition_matrix(self):
        """Create transition matrix"""
        print("Creating transition matrix...")
        adj_matrix = nx.to_scipy_sparse_array(self.graph)
        degree_vector = sp.csr_matrix(1 / np.sum(adj_matrix, axis=0))
        self.transition_matrix = adj_matrix.multiply(degree_vector).T
        self.graph = None  # Free memory

    def _create_transition_dict(self):
        """Convert matrix to dictionary form"""
        print("Creating transition dictionary...")
        rows, cols = self.transition_matrix.nonzero()
        
        prev_row = -1
        for row, col in zip(rows, cols):
            if row != prev_row:
                self.transition_dict.setdefault(row, {'product': [], 'probability': []})
            self.transition_dict[row]['product'].append(col)
            self.transition_dict[row]['probability'].append(self.transition_matrix[row, col])
            prev_row = row
        
        self.transition_matrix = None  # Free memory

    def _generate_random_walks(self):
        """Generate sequences using random walk"""
        print("Generating random walks...")
        n_nodes = len(self.node_dict)
        self.sample_array = np.zeros((n_nodes * self.num_sequence, self.sequence_len), dtype=int)
        
        random.seed(42)
        for node_idx in range(n_nodes):
            if node_idx % 1000 == 0:
                print(f"Processing node {node_idx}/{n_nodes}")
            for seq_idx in range(self.num_sequence):
                current_node = node_idx
                for step in range(self.sequence_len):
                    self.sample_array[node_idx*self.num_sequence + seq_idx, step] = current_node
                    current_node = random.choices(
                        self.transition_dict[current_node]['product'],
                        weights=self.transition_dict[current_node]['probability'],
                        k=1
                    )[0]
        
        # Map back to original IDs
        self.sample_array = np.vectorize(self.node_dict.get)(self.sample_array)

    def _save_sequences_to_txt(self):
        """Save sequences to txt file"""
        print(f"Saving sequences to {self.output_path}")
        with open(self.output_path, 'w') as f:
            for seq in self.sample_array:
                f.write(' '.join(seq) + '\n')

    def generate_sequences(self, isTrain=True):
        """Main pipeline to generate sequences"""
        input_file = self._group_history_transaction(isTrain)
        
        edgelist_path = self._get_frequency_pairs(input_file, isTrain)
        
        self._build_graph(edgelist_path)
        
        self._create_transition_matrix()
        
        self._create_transition_dict()
        
        self._generate_random_walks()
        
        self._save_sequences_to_txt()
        
        print("\nFinal Statistics:")
        print(f"Number of sequences: {len(self.sample_array)}")
        print(f"Sequence length: {self.sequence_len}")
        print(f"Unique products: {len(np.unique(self.sample_array))}")
        print(f"Example sequence[0]: {' '.join(self.sample_array[0])}")

def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):

        # uid = np.random.randint(1, usernum + 1)
        while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()