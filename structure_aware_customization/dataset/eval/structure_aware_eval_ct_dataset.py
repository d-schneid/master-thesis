from structure_aware_customization.dataset.structure_aware_ct_dataset import StructureAwareCTDataset

import torch


class StructureAwareEvalCTDataset(StructureAwareCTDataset):

	def __getitem__(self, idx):
		sample = super().__getitem__(idx)
		start_completion_idx = 0

		sample['loss_mask'].zero_()
		sample['loss_mask'][start_completion_idx] = 1
		sample['text_token_ids'][start_completion_idx + 1 :] = 0

		sample['text_token_rel_pos_ids'][start_completion_idx + 1 :, :] = 0
		sample['text_token_rel_pos_ids'][:, start_completion_idx + 1 :] = 0

		sample['attn_text_tokens'][start_completion_idx + 1 :, :] = -1e9
		sample['attn_text_tokens'][:, start_completion_idx + 1 :] = -1e9

		return sample

	def build_attn_bias(self, batch_dict, first_col_matrix, second_col_matrix, third_col_matrix):
		# individual padded attention masks
		attn_text_tokens = batch_dict['attn_text_tokens']
		attn_code_text = batch_dict['attn_code_text']
		attn_ast_text = batch_dict['attn_ast_text']
		attn_dfg_text = batch_dict['attn_dfg_text']

		# Compute transpose
		attn_code_text_T = attn_code_text.transpose(1, 2).clone()
		attn_code_text_T[:, 0, :] = 0

		attn_ast_text_T = attn_ast_text.transpose(1, 2).clone()
		attn_ast_text_T[:, 0, :] = 0

		attn_dfg_text_T = attn_dfg_text.transpose(1, 2).clone()
		attn_dfg_text_T[:, 0, :] = 0

		# Build block matrices column-wise
		first_col_matrix = torch.cat((first_col_matrix, attn_ast_text_T), dim=1)
		second_col_matrix = torch.cat((second_col_matrix, attn_dfg_text_T), dim=1)
		third_col_matrix = torch.cat((third_col_matrix, attn_code_text_T), dim=1)
		fourth_col_matrix = torch.cat((attn_ast_text, attn_dfg_text, attn_code_text, attn_text_tokens), dim=1)

		attn_bias = torch.cat((first_col_matrix, second_col_matrix, third_col_matrix, fourth_col_matrix), dim=2)

		return attn_bias
