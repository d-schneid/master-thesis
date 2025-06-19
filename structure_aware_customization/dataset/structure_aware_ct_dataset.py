from structure_aware_customization.dataset.structure_aware_dataset import StructureAwareDataset
from data_preprocessing.datasets.dataset import Dataset as EncapsulatedDataset

import torch


class StructureAwareCTDataset(StructureAwareDataset):

	def __init__(self, dataset: EncapsulatedDataset) -> None:
		super().__init__(dataset=dataset)

	def __getitem__(self, idx):
		batch = super().__getitem__(idx)

		text_tokens = batch['text_token_ids']
		labels = torch.cat([torch.tensor([self.padding_value], dtype=text_tokens.dtype), text_tokens[1:]])
		loss_mask = torch.cat([torch.tensor([0], dtype=text_tokens.dtype), torch.ones(len(text_tokens[:-1]), dtype=text_tokens.dtype)])

		batch['labels'] = labels
		batch['loss_mask'] = loss_mask

		return batch

	def get_key_not_in(self):
		return ['code_token_ids', 'text_token_ids', 'dfg_node_mask', 'lr_paths_len', 'labels', 'loss_mask']

	def get_attn_keys(self):
		return ['attn_code_tokens', 'attn_text_tokens', 'attn_ast_leaves', 'attn_dfg_edges', 'attn_code_ast',
				'attn_code_dfg', 'attn_code_text', 'attn_ast_text', 'attn_dfg_text']

	def get_labels_loss_pad_len(self, batch_dict):
		return batch_dict['dfg_node_mask'][0].size(0) + batch_dict['lr_paths_len'][0].size(0) + batch_dict['code_token_ids'][0].size(0)

	def build_attn_bias(self, batch_dict, first_col_matrix, second_col_matrix, third_col_matrix):
		# individual padded attention masks
		attn_text_tokens = batch_dict['attn_text_tokens']
		attn_code_text = batch_dict['attn_code_text']
		attn_ast_text = batch_dict['attn_ast_text']
		attn_dfg_text = batch_dict['attn_dfg_text']

		# Compute transpose
		attn_code_text_T_shape = attn_code_text.transpose(1, 2).shape
		attn_code_text_T = torch.full(attn_code_text_T_shape, fill_value=0)

		attn_ast_text_T_shape = attn_ast_text.transpose(1, 2).shape
		attn_ast_text_T = torch.full(attn_ast_text_T_shape, fill_value=0)

		attn_dfg_text_T_shape = attn_dfg_text.transpose(1, 2).shape
		attn_dfg_text_T = torch.full(attn_dfg_text_T_shape, fill_value=0)

		# Build block matrices column-wise
		first_col_matrix = torch.cat((first_col_matrix, attn_ast_text_T), dim=1)
		second_col_matrix = torch.cat((second_col_matrix, attn_dfg_text_T), dim=1)
		third_col_matrix = torch.cat((third_col_matrix, attn_code_text_T), dim=1)
		fourth_col_matrix = torch.cat((attn_ast_text, attn_dfg_text, attn_code_text, attn_text_tokens), dim=1)

		attn_bias = torch.cat((first_col_matrix, second_col_matrix, third_col_matrix, fourth_col_matrix), dim=2)

		return attn_bias
