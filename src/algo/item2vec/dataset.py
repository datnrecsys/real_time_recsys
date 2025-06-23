import json
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.distributed import get_rank, get_world_size
from torch.utils.data import IterableDataset, get_worker_info
from tqdm.auto import tqdm
import pandas as pd
from typing import Optional

class SkipGramDataset(IterableDataset):

	'''
	IterableDataset for SkipGram model
	'''

	def __init__(
		self,
		train_interaction_df: pd.DataFrame,
		val_interaction_df: Optional[pd.DataFrame] = None,
		user_col: str = "user_id",
		item_col: str = "parent_asin",
		window_size: int = 2,
		negative_samples: int = 5,
		item_id_to_idx: dict = None,
		ddp: bool = False,
		mode: str = "train",

	):
		'''
		Args:
		interaction_df: Pandas dataframe containing the interactions data (user id and item id are required)
		interacted_dict (defaultdict(set)): A dictionary that keeps track of the other items that shared the same basket with the target item. Those items are ignored when negative sampling.
		item_freq (defaultdict(int)): A dictionary that keeps track the item frequency. It's used to
		window_size (int): The context window size.
		negative_samples (int): Number of negative samples for each positive pair.
		item_id_to_idx (dict): Mapper between item id (string) to item index (int) (Assumption: starting from 0)
		ddp (bool): whether we're using DDP for distributed training or not

	The reason that interacted_dict and item_freq can be passed into the initialization is that at val dataset creation we want to do negative sampling based on the data from the train set as well.
		'''

		assert user_col in train_interaction_df.columns and item_col in train_interaction_df.columns, "user_id and item_id columns are required in train_interaction_df"
		assert val_interaction_df is not None if mode == "val" else "do nothing", "val_interaction_df is required for validation mode"
		self.window_size = window_size
		self.negative_samples = negative_samples
		self.ddp = ddp
		self.interacted = defaultdict(set)     # Keep track of which item-pair co-occur in one basket(e.g. {item1: {item1, item2, item3}})
		self.item_freq = defaultdict(int)      # Keep track of the frequency of each item
		self.mode = mode

		if item_id_to_idx is None:
			self.item_id_to_idx = dict()
			self.item_idx_to_id = dict()
		else:
			self.item_id_to_idx = item_id_to_idx
			self.item_idx_to_id = {v: k for k, v in item_id_to_idx.items()}

		self.num_targets = 0  # Number of item in all sequences

		#  Keep tracked of which item-pair co-occur in one basket
		# When doing negative sampling we do not consider the other items that the target item has shared basket
		logger.info("Processing sequences...")

		# Group by customer_id and get the list of items that user interacted with
		train_sequence_df = train_interaction_df.groupby(user_col, as_index= False)[item_col].\
																	apply(lambda x: list(set(x))).\
																	loc[lambda df: df[item_col].map(len) > 1]
		# print(train_sequence_df)
		self.train_sequences = train_sequence_df[item_col]
		# print(self.train_sequences)
		
		if val_interaction_df is not None:
			val_sequence_df = val_interaction_df.groupby(user_col, as_index= False)[item_col].\
																	apply(lambda x: list(set(x))).\
																	loc[lambda df: df[item_col].map(len) > 1]
			self.val_sequences = val_sequence_df[item_col]
		for seq in self.train_sequences:  # Iterate through each sequence list

			# Construct id_to_idx and idx_to_id mapping (if needed)
			for item in seq:
				item = str(item)
				idx = self.item_id_to_idx.get(item)
				if idx is None:
					idx = len(self.item_id_to_idx)
					self.item_id_to_idx[item] = idx
					self.item_idx_to_id[idx] = item
				self.num_targets += 1
			seq_idx_set = set([self.item_id_to_idx[str(item)] for item in seq])
			for idx in seq_idx_set:
				# An item can be considered that it has interacted with itself
				# This helps with negative sampling later
				self.interacted[idx].update(seq_idx_set)   # immutable --> can update directly
				self.item_freq[idx] += 1
		if item_id_to_idx is None:
			self.vocab_size = len(self.item_freq)
		else:
			self.vocab_size = len(item_id_to_idx)

		items_idx, frequencies = zip(*self.item_freq.items())
		self.item_freq_array = np.zeros(self.vocab_size, dtype=np.float32)
		self.item_freq_array[np.array(items_idx)] = frequencies
		self.items = np.arange(self.vocab_size)

		# Use a smoothed frequency distribution for negative sampling
		# The smoothing factor (0.75) can be tuned
		self.sampling_probs = self.item_freq_array**0.75
		self.sampling_probs /= self.sampling_probs.sum()

	def get_process_info(self):
		"""
		Get information about which process is processing the data so that we can correctly split up the data based on iteration
		"""
		if not self.ddp:
			num_replicas = 1
			rank = 0
			return num_replicas, rank

		worker_info = get_worker_info()
		# number of workers in each process (e.g. 1 process has 2 workers)
		num_workers = worker_info.num_workers if worker_info is not None else 1    
		# id of the worker within the process (e.g. Process 1 has worker with id 0 and id 1)
		worker_id = worker_info.id if worker_info is not None else 0
		
		# Total number of processes (e.g. We want to train on 2 GPUs --> world_size = 2)
		world_size = get_world_size()
		# the rank of the current process 
		process_rank = get_rank()

		# Total number of workers across all processes
		num_replicas = num_workers * world_size
		rank = process_rank * num_workers + worker_id

		return num_replicas, rank

	def __iter__(self):
		num_replicas, rank = self.get_process_info()
		idx = 0
		if self.mode == "train":
			for seq in self.train_sequences:
				# print(seq)
				for i in range(len(seq)):
					if idx % num_replicas != rank:
						idx += 1
						continue

					yield self._get_item(seq, i)
					idx += 1
		elif self.mode == "val":
			for seq in self.val_sequences:
				# print("seq: ", seq)
				for i in range(len(seq)):
					if idx % num_replicas != rank:
						idx += 1
						continue

					yield self._get_item(seq, i)
					idx += 1
		else:
			raise ValueError("mode should be either 'train' or 'val'")

	def _get_item(self, seq, i):
		# Convert list of item_id to a list of item_idx
		sequence = [self.item_id_to_idx[str(item)] for item in seq]
		target_item = sequence[i]

		positive_pairs = []
		labels = []

		start = max(i-self.window_size, 0)
		end = min(i+self.window_size+1, len(sequence))

		for j in range(start, end):
			if j == i:
				continue
			positive_pairs.append((target_item, sequence[j]))
			labels.append(1)

		negative_pairs = []

		# print('positive_pairs: ', positive_pairs)
		for target_item, _ in positive_pairs:
			# Mask out the items that the target item has interacted with
			# Then sample the remaining items based on the item frequency as negative items
			negative_sampling_probs = deepcopy(self.sampling_probs)
			negative_sampling_probs[list(self.interacted[target_item])] = 0
			if negative_sampling_probs.sum() == 0:
				# This target_item has interacted with every other items
				negative_sampling_probs = np.ones(len(negative_sampling_probs))

			negative_sampling_probs /= negative_sampling_probs.sum()

			negative_items = np.random.choice(
				self.items,
				size=self.negative_samples,
				p=negative_sampling_probs,
				replace=False,
			)

			for negative_item in negative_items:
				negative_pairs.append((target_item, negative_item))
				labels.append(0)

		pairs = positive_pairs + negative_pairs

		target_items = torch.tensor([pair[0] for pair in pairs], dtype=torch.long)
		context_items = torch.tensor([pair[1] for pair in pairs], dtype=torch.long)
		labels = torch.tensor(labels, dtype=torch.float)
		# print(target_items, context_items, labels)
		return {
			"target_items": target_items,
			"context_items": context_items,
			"labels": labels,
		}
		
	def collate_fn(self, batch):
		target_items = []
		context_items = []
		labels = []
		# print(batch)
		for record in batch:
			if record:
				target_items.append(record["target_items"])
				context_items.append(record["context_items"])
				labels.append(record["labels"])
		return {
			"target_items": torch.cat(target_items, dim=0),
			"context_items": torch.cat(context_items, dim=0),
			"labels": torch.cat(labels, dim=0),
		}

	def save_id_mappings(self, filepath: str):
		with open(filepath, "w") as f:
			json.dump(
				{
					"id_to_idx": self.item_id_to_idx,
					"idx_to_id": self.item_idx_to_id,
				},
				f,
			)

	@classmethod
	def get_default_loss_fn(cls):
		loss_fn = nn.BCELoss()
		return loss_fn

	@classmethod
	def forward(cls, model, batch_input, loss_fn=None, device="cpu"):
		predictions = model.predict_train_batch(batch_input, device=device).squeeze()
		labels = batch_input["labels"].float().to(device).squeeze()

		if loss_fn is None:
			loss_fn = cls.get_default_loss_fn()

		loss = loss_fn(predictions, labels)
		return loss