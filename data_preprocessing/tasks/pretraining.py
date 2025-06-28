from data_preprocessing.tasks.task import Task

import numpy as np


class Pretraining(Task):

	def __init__(self):
		super().__init__(task='pretraining')

	def get_cols(self):
		return [
			'attn_code_tokens',
			'attn_ast_leaves',
			'attn_dfg_edges',
			'attn_code_ast',
			'attn_code_dfg',
		]

	def _generate_sample(self, row):
		return {}

	def _get_1d_features(self):
		return []

	def _get_2d_features(self):
		return []

	def generate_adj_matrix(self, edges, num_nodes):
		adj_matrix = np.full((num_nodes, num_nodes), self.attn_bias_ignore, dtype=np.float32)

		for to_node, from_nodes in edges:
			for from_node in from_nodes:
				if from_node <= to_node: # account for masked attention in pretraining
					adj_matrix[to_node, from_node] = self.attn_bias_attend

		return adj_matrix

	def compute_attention_masks(self, data):
		data['attn_code_tokens'] = data['code_tokens'].apply(self.masked_attention)
		data['attn_ast_leaves'] = data['lr_paths_len'].apply(self.masked_attention)
		data['attn_dfg_edges'] = data.apply(
			lambda row: self.generate_adj_matrix(row['dfg_edges'], len(row['dfg_node_mask'])),
			axis=1
		)

		data['attn_code_ast'] = data.apply(
			lambda row: self.build_attention_matrix(
				row=row,
				attn_col='ast_leaf_code_token_idxs',
				num_targets=len(row['lr_paths_len']),
				attn_col_offset=1  # adjust for padding of AST leaves
			),
			axis=1
		)

		data['attn_code_dfg'] = data.apply(
			lambda row: self.build_attention_matrix(
				row=row,
				attn_col='dfg_node_code_token_idxs',
				num_targets=len(row['dfg_node_mask']),
				attn_col_offset=1  # adjust for padding of DFG nodes
			),
			axis=1
		)

		return data
