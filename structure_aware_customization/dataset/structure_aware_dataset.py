import json
from abc import ABC, abstractmethod

from data_preprocessing.data_handler import DataHandler, PAD_TOK_ID_DFG
from data_preprocessing.datasets.dataset import Dataset as AbstractDataset

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import h5py
import numpy as np


class StructureAwareDataset(ABC, Dataset):

	def __init__(self, dataset: AbstractDataset) -> None:
		super().__init__()
		self.data_handler = DataHandler(dataset=dataset)
		self.max_seq_len = dataset.task.max_seq_len
		self.padding_value = self.data_handler.tokenizer.eos_token_id
		self.h5_file = h5py.File(dataset.h5_path, 'r')

		with open(dataset.num_samples_path, 'r') as f:
			metadata = json.load(f)
		self.num_samples = metadata['num_samples']

		with open(dataset.metadata_path_pretraining, 'r') as f:
			self.metadata = json.load(f)
		self.pad_tok_id_ast = self.metadata['num_ast_node_types']

		self.numpy_to_torch_dtype = {
			np.uint8: torch.int32,
			np.uint16: torch.int32,
			np.float16: torch.bfloat16,
			np.float32: torch.bfloat16,
		}

	def __len__(self) -> int:
		return self.num_samples

	def __getitem__(self, idx):
		key = f'sample_{idx}'
		group = self.h5_file[key]
		sample = {
			name: torch.from_numpy(dataset[()]).to(self.numpy_to_torch_dtype[dataset[()].dtype.type])
			for name, dataset in group.items()
		}

		return sample

	def __del__(self):
		if hasattr(self, 'h5_file') and self.h5_file:
			self.h5_file.close()

	@abstractmethod
	def get_1d_keys(self):
		pass

	@abstractmethod
	def get_attn_keys(self):
		pass

	@abstractmethod
	def get_labels_loss_pad_len(self, batch_dict):
		pass

	@abstractmethod
	def build_attn_bias(self, batch_dict, first_col_matrix, second_col_matrix, third_col_matrix):
		pass

	@abstractmethod
	def get_2d_tokens_for_max_seq_len_padding(self):
		pass

	def get_1d_tokens_for_max_seq_len_padding(self):
		return ['labels', 'loss_mask']

	def pad_batch_to_max_seq_len(self, batch_dict):
		# append padding value to code tokens to pad sequence
		num_tokens_before = self.get_labels_loss_pad_len(batch_dict)

		for key in self.get_1d_tokens_for_max_seq_len_padding():
			num_pad_right_tokens = self.max_seq_len - num_tokens_before - batch_dict[key][0].size(0)
			num_pad_left_tokens = 0
			if key in ['labels', 'loss_mask']:
				num_pad_left_tokens = num_tokens_before
			batch_dict[key] = F.pad(batch_dict[key], (num_pad_left_tokens, num_pad_right_tokens), value=self.padding_value)

		for key in self.get_2d_tokens_for_max_seq_len_padding():
			max_rows = self.max_seq_len - num_tokens_before
			max_cols = self.max_seq_len - num_tokens_before
			padding_value = self.data_handler.task.attn_bias_ignore

			if key in ['attn_code_tokens', 'attn_text_tokens', 'code_token_rel_pos_ids', 'text_token_rel_pos_ids']:
				if key in ['code_token_rel_pos_ids', 'text_token_rel_pos_ids']:
					padding_value = self.padding_value
				batch_dict[key] = torch.stack(pad_2d_tensors(batch_dict[key], padding_value=padding_value,
															 max_rows=max_rows, max_cols=max_cols))
			elif key in ['attn_code_ast', 'attn_code_dfg']:
				keep_num_cols = batch_dict[key].shape[2]
				batch_dict[key] = torch.stack(pad_2d_tensors(batch_dict[key], padding_value=padding_value,
															 max_rows=max_rows, max_cols=keep_num_cols))
			elif key in ['attn_code_text', 'attn_ast_text', 'attn_dfg_text']:
				keep_num_rows = batch_dict[key].shape[1]
				batch_dict[key] = torch.stack(pad_2d_tensors(batch_dict[key], padding_value=padding_value,
															 max_rows=keep_num_rows, max_cols=max_cols))

	def pad_within_batch(self, batch, batch_dict):
		for key in batch[0].keys():
			batch_dict[key] = [sample[key] for sample in batch]
			if key not in self.get_1d_keys():
				if key == 'lr_paths_types':
					batch_dict[key] = pad_2d_tensors(batch_dict[key], padding_value=self.pad_tok_id_ast)
				elif key in self.get_attn_keys():
					batch_dict[key] = pad_2d_tensors(batch_dict[key], padding_value=self.data_handler.task.attn_bias_ignore)
				else:
					batch_dict[key] = pad_2d_tensors(batch_dict[key], padding_value=self.padding_value)

			padding_value = self.padding_value
			if key == 'dfg_node_mask':
				padding_value = PAD_TOK_ID_DFG
			if key in self.get_attn_keys():
				padding_value = self.data_handler.task.attn_bias_ignore
			batch_dict[key] = pad_sequence(batch_dict[key], batch_first=True, padding_value=padding_value)

	def build_attn_mask(self, batch_dict):
		# individual padded attention masks
		attn_code_tokens = batch_dict['attn_code_tokens']
		attn_ast_leaves = batch_dict['attn_ast_leaves']
		attn_dfg_edges = batch_dict['attn_dfg_edges']
		attn_code_ast = batch_dict['attn_code_ast']
		attn_code_dfg = batch_dict['attn_code_dfg']

		# Compute transpose
		attn_code_ast_T = attn_code_ast.transpose(1, 2)
		attn_code_dfg_T = attn_code_dfg.transpose(1, 2)

		# Compute null matrix for attention between AST leaves and DFG edges
		attn_ast_dfg = torch.full((attn_ast_leaves.size(0), attn_ast_leaves.size(1), attn_dfg_edges.size(2)),
								  fill_value=self.data_handler.task.attn_bias_ignore)
		attn_ast_dfg_T = attn_ast_dfg.transpose(1, 2)

		# Build block matrices column-wise
		first_col_matrix = torch.cat((attn_ast_leaves, attn_ast_dfg_T, attn_code_ast), dim=1)
		second_col_matrix = torch.cat((attn_ast_dfg, attn_dfg_edges, attn_code_dfg), dim=1)
		third_col_matrix = torch.cat((attn_code_ast_T, attn_code_dfg_T, attn_code_tokens), dim=1)

		attn_bias = self.build_attn_bias(batch_dict, first_col_matrix, second_col_matrix, third_col_matrix)

		batch_dict['attention_bias'] = attn_bias.unsqueeze(1).bfloat16() # broadcast across all attention heads

		keys_to_remove = self.get_attn_keys()
		for key in keys_to_remove:
			del batch_dict[key]

	def prepare_leaf_embedding(self, batch_dict):
		batch_size = batch_dict['lr_paths_types'].shape[0]
		len_longest_lr_path = batch_dict['lr_paths_types'].shape[-1]

		ast_node_heights = torch.arange(len_longest_lr_path, dtype=torch.int32).unsqueeze(0).repeat(batch_size, 1).unsqueeze(1)
		batch_dict['ast_node_heights'] = ast_node_heights

		len_lr_path_range = torch.arange(len_longest_lr_path).view(1, 1, len_longest_lr_path)
		lr_paths_len_mask = len_lr_path_range < batch_dict['lr_paths_len'].unsqueeze(-1)
		batch_dict['lr_paths_len_mask'] = lr_paths_len_mask

		del batch_dict['lr_paths_len']

	def collate_fn(self, batch):
		# Initialize a dictionary to store the batch data
		batch_dict = {}
		self.pad_within_batch(batch, batch_dict)
		self.pad_batch_to_max_seq_len(batch_dict)
		self.build_attn_mask(batch_dict)
		self.prepare_leaf_embedding(batch_dict)

		return batch_dict


def pad_2d_tensors(tensor_list, padding_value, padding_side='right', max_rows=None, max_cols=None):
	if max_rows is None or max_cols is None:
		max_rows = max(tensor.size(0) for tensor in tensor_list)
		max_cols = max(tensor.size(1) for tensor in tensor_list)

	padded_tensors = []
	for tensor in tensor_list:
		rows_to_pad = max_rows - tensor.size(0)
		cols_to_pad = max_cols - tensor.size(1)

		if padding_side == 'right':
			padded_tensor = torch.nn.functional.pad(tensor, (0, cols_to_pad, 0, rows_to_pad), mode='constant', value=padding_value)
		else:
			padded_tensor = torch.nn.functional.pad(tensor, (cols_to_pad, 0, 0, rows_to_pad), mode='constant', value=padding_value)

		padded_tensors.append(padded_tensor)

	return padded_tensors
