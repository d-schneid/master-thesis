from structure_aware_customization.dataset.structure_aware_pretraining_dataset import StructureAwarePretrainingDataset

import torch


class StructureAwareCCDataset(StructureAwarePretrainingDataset):

	def __getitem__(self, idx):
		sample = super().__getitem__(idx)

		start_completion_idx = sample['start_completion_idx'].item()
		sample['loss_mask'][:start_completion_idx] = 0
		del sample['start_completion_idx']

		sample['attn_code_ast_all'] = sample['attn_code_ast'].clone()
		sample['attn_code_ast_all'][start_completion_idx + 1 :] = 0

		sample['attn_code_dfg_all'] = sample['attn_code_dfg'].clone()
		sample['attn_code_dfg_all'][start_completion_idx + 1 :] = 0

		return sample

	def get_2d_tokens_for_max_seq_len_padding(self):
		return ['attn_code_tokens', 'attn_code_ast', 'attn_code_dfg', 'code_token_rel_pos_ids', 'attn_code_ast_all', 'attn_code_dfg_all']

	def get_attn_keys(self):
		return ['attn_code_tokens', 'attn_ast_leaves', 'attn_dfg_edges', 'attn_code_ast', 'attn_code_ast_all', 'attn_code_dfg', 'attn_code_dfg_all']

	def build_attn_mask(self, batch_dict):
		# individual padded attention masks
		attn_code_tokens = batch_dict['attn_code_tokens']
		attn_ast_leaves = batch_dict['attn_ast_leaves']
		attn_dfg_edges = batch_dict['attn_dfg_edges']
		attn_code_ast = batch_dict['attn_code_ast']
		attn_code_ast_all = batch_dict['attn_code_ast_all']
		attn_code_dfg = batch_dict['attn_code_dfg']
		attn_code_dfg_all = batch_dict['attn_code_dfg_all']

		# Compute transpose
		attn_code_ast_T = attn_code_ast.transpose(1, 2)
		attn_code_dfg_T = attn_code_dfg.transpose(1, 2)

		# Compute null matrix for attention between AST leaves and DFG edges
		attn_ast_dfg = torch.full((attn_ast_leaves.size(0), attn_ast_leaves.size(1), attn_dfg_edges.size(2)),
								  fill_value=self.data_handler.task.attn_bias_ignore)
		attn_ast_dfg_T = attn_ast_dfg.transpose(1, 2)

		# Build block matrices column-wise
		first_col_matrix = torch.cat((attn_ast_leaves, attn_ast_dfg_T, attn_code_ast_all), dim=1)
		second_col_matrix = torch.cat((attn_ast_dfg, attn_dfg_edges, attn_code_dfg_all), dim=1)
		third_col_matrix = torch.cat((attn_code_ast_T, attn_code_dfg_T, attn_code_tokens), dim=1)

		attn_bias = self.build_attn_bias(batch_dict, first_col_matrix, second_col_matrix, third_col_matrix)

		batch_dict['attention_bias'] = attn_bias.unsqueeze(1).bfloat16() # broadcast across all attention heads

		keys_to_remove = self.get_attn_keys()
		for key in keys_to_remove:
			del batch_dict[key]
