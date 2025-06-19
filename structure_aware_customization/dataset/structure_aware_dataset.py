import json
from abc import ABC, abstractmethod

from data_preprocessing.data_handler import DataHandler, PAD_TOK_ID_DFG
from data_preprocessing.datasets.dataset import Dataset as EncapsulatedDataset

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import h5py


class StructureAwareDataset(ABC, Dataset):

	def __init__(self, dataset: EncapsulatedDataset) -> None:
		super().__init__()
		self.data_handler = DataHandler(dataset=dataset)
		self.padding_value = self.data_handler.tokenizer.eos_token_id
		self.h5_file = h5py.File(dataset.h5_path, 'r')

		with open(dataset.metadata_path, 'r') as f:
			metadata = json.load(f)
		self.num_samples = metadata['num_samples']

		with open(dataset.metadata_path_train, 'r') as f:
			self.metadata = json.load(f)
		self.pad_tok_id_ast = self.metadata['num_ast_node_types']

		self.data = []
		for key in self.h5_file.keys():
			group = self.h5_file[key]
			sample = {
				name: torch.from_numpy(dataset[()]) for name, dataset in group.items()
			}
			self.data.append(sample)

		self.h5_file.close()

	def __len__(self) -> int:
		return self.num_samples

	def __getitem__(self, idx):
		return self.data[idx]

	@abstractmethod
	def get_key_not_in(self):
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

	def collate_fn(self, batch):
		# Initialize a dictionary to store the batch data
		batch_dict = {}
		for key in batch[0].keys():
			batch_dict[key] = [sample[key] for sample in batch]
			if key not in self.get_key_not_in():
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

			if key in ['labels', 'loss_mask']:
				batch_dict[key] = pad_sequence(batch_dict[key], batch_first=True, padding_value=padding_value, padding_side='left')
			else:
				batch_dict[key] = pad_sequence(batch_dict[key], batch_first=True, padding_value=padding_value)

		pad_len = self.get_labels_loss_pad_len(batch_dict)
		batch_dict = pad_labels_loss_mask(batch_dict, pad_len)

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

		return batch_dict


def pad_labels_loss_mask(batch_dict, pad_len):
	labels = batch_dict['labels']
	loss_mask = batch_dict['loss_mask']

	padded_labels = []
	padded_loss_mask = []

	for label, mask in zip(labels, loss_mask):
		padded_label = F.pad(label, (pad_len, 0), value=0)
		padded_mask = F.pad(mask, (pad_len, 0), value=0)
		padded_labels.append(padded_label)
		padded_loss_mask.append(padded_mask)

	batch_dict['labels'] = torch.stack(padded_labels)
	batch_dict['loss_mask'] = torch.stack(padded_loss_mask)

	return batch_dict


def pad_2d_tensors(tensor_list, padding_value, padding_side='right'):
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


def pad_inner_lists(list_of_lists, padding_value, padding_side='right'):
	tensors = [torch.tensor(x) for x in list_of_lists]

	return pad_sequence(tensors, batch_first=True, padding_value=padding_value, padding_side=padding_side) if tensors else [torch.tensor(-1)]
