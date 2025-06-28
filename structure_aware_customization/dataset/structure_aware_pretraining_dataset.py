from structure_aware_customization.dataset.structure_aware_dataset import StructureAwareDataset
from data_preprocessing.datasets.dataset import Dataset

import torch


class StructureAwarePretrainingDataset(StructureAwareDataset):

	def __init__(self, dataset: Dataset) -> None:
		super().__init__(dataset=dataset)

	def __getitem__(self, idx):
		batch = super().__getitem__(idx)

		code_tokens = batch['code_token_ids']
		labels = torch.cat([code_tokens[1:], torch.tensor([self.padding_value], dtype=code_tokens.dtype)])
		loss_mask = torch.cat([torch.ones(len(code_tokens[:-1]), dtype=code_tokens.dtype), torch.tensor([0], dtype=code_tokens.dtype)])

		batch['labels'] = labels
		batch['loss_mask'] = loss_mask

		return batch

	def get_2d_tokens_for_max_seq_len_padding(self):
		return ['attn_code_tokens', 'attn_code_ast', 'attn_code_dfg', 'code_token_rel_pos_ids']

	def get_1d_tokens_for_max_seq_len_padding(self):
		return super().get_1d_tokens_for_max_seq_len_padding() + ['code_token_ids']

	def get_1d_keys(self):
		return ['code_token_ids', 'dfg_node_mask', 'lr_paths_len', 'labels', 'loss_mask']

	def get_attn_keys(self):
		return ['attn_code_tokens', 'attn_ast_leaves', 'attn_dfg_edges', 'attn_code_ast', 'attn_code_dfg']

	def get_labels_loss_pad_len(self, batch_dict):
		return batch_dict['dfg_node_mask'][0].size(0) + batch_dict['lr_paths_len'][0].size(0)

	def build_attn_bias(self, batch_dict, first_col_matrix, second_col_matrix, third_col_matrix):
		# Build block matrices column-wise
		attn_bias = torch.cat((first_col_matrix, second_col_matrix, third_col_matrix), dim=2)

		return attn_bias
