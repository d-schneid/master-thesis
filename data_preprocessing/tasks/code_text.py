from data_preprocessing.tasks.task import Task

import numpy as np
import torch


class CodeText(Task):

	def __init__(self):
		super().__init__(task='code_text')

	def get_cols(self):
		return [
			'text_tokens',
			'text_tokens_rel_pos_ids',
			'attn_text_tokens',
			'attn_code_tokens',
			'attn_ast_leaves',
			'attn_dfg_edges',
			'attn_code_ast',
			'attn_code_dfg',
			'attn_code_text',
			'attn_ast_text',
			'attn_dfg_text',
		]

	def _generate_sample(self, row):
		update = {
			"text_token_ids": row["text_tokens"],
			"text_token_rel_pos_ids": row["text_tokens_rel_pos_ids"],
			"attn_text_tokens": row["attn_text_tokens"],
			"attn_code_text": row["attn_code_text"],
			"attn_ast_text": row["attn_ast_text"],
			"attn_dfg_text": row["attn_dfg_text"],
		}

		return update

	def generate_adj_matrix(self, edges, num_nodes):
		adj_matrix = np.full((num_nodes, num_nodes), self.attn_bias_ignore, dtype=np.float32)

		for to_node, from_nodes in edges:
			for from_node in from_nodes:
				adj_matrix[to_node, from_node] = self.attn_bias_attend

		return adj_matrix

	def full_attention(self, row, row_len):
		col_len = len(row['text_tokens'])

		return np.full((row_len, col_len), self.attn_bias_ignore, dtype=np.float32)

	def compute_attention_masks(self, data):
		data['attn_text_tokens'] = data['text_tokens'].apply(self.masked_attention)
		data['attn_code_tokens'] = data['code_tokens'].apply(lambda row: np.zeros((len(row), len(row)), dtype=np.float32))
		data['attn_ast_leaves'] = data['lr_paths_len'].apply(lambda row: np.zeros((len(row), len(row)), dtype=np.float32))
		data['attn_dfg_edges'] = data.apply(
			lambda row: self.generate_adj_matrix(row['dfg_edges'], len(row['dfg_node_mask'])
			),
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

		data['attn_code_text'] = data.apply(
			lambda row: self.full_attention(
				row=row,
				row_len=len(row['code_tokens'])
			),
			axis=1
		)

		data['attn_ast_text'] = data.apply(
			lambda row: self.full_attention(
				row=row,
				row_len=len(row['lr_paths_len'])
			),
			axis=1
		)

		data['attn_dfg_text'] = data.apply(
			lambda row: self.full_attention(
				row=row,
				row_len=len(row['dfg_node_mask'])
			),
			axis=1
		)

		return data
