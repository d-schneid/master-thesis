from abc import ABC, abstractmethod

import numpy as np


class Task(ABC):

	def __init__(self, task):
		self.task = task
		self.attn_bias_attend = 0
		self.attn_bias_ignore = -1e9
		self.max_seq_len = 1024

	@abstractmethod
	def decode(self, logits, batch_no_labels, batch):
		pass

	@abstractmethod
	def compute_attention_masks(self, data):
		pass

	@abstractmethod
	def get_cols(self):
		pass

	@abstractmethod
	def _generate_sample(self, row):
		pass

	def generate_sample(self, row):
		sample = {
			"code_token_ids": row["code_tokens"],
			"code_token_rel_pos_ids": row["code_tokens_rel_pos_ids"],
			"ll_sims": row["ll_sims"],
			"lr_paths_types": row["lr_paths_types"],
			"lr_paths_len": row["lr_paths_len"],
			"dfg_node_mask": row["dfg_node_mask"],
			"attn_code_tokens": row["attn_code_tokens"],
			"attn_ast_leaves": row["attn_ast_leaves"],
			"attn_dfg_edges": row["attn_dfg_edges"],
			"attn_code_ast": row["attn_code_ast"],
			"attn_code_dfg": row["attn_code_dfg"],
		}
		sample.update(self._generate_sample(row))

		return sample

	def get_max_seq_len_cols(self):
		return ["code_tokens", "lr_paths_len", "dfg_node_mask"]

	def filter_max_seq_len(self, data):
		cols = self.get_max_seq_len_cols()
		length_sums = data.apply(lambda row: sum(len(row[col]) for col in cols), axis=1)
		data_filtered = data[length_sums <= self.max_seq_len].reset_index(drop=True)

		return data_filtered

	@abstractmethod
	def _get_1d_features(self):
		pass

	def get_1d_features(self):
		features = [("code_token_ids", np.int32), ("lr_paths_len", np.int32), ("dfg_node_mask", np.int32)]

		return features + self._get_1d_features()

	@abstractmethod
	def _get_2d_features(self):
		pass

	def get_2d_features(self):
		features = [("code_token_rel_pos_ids", np.int32), ("ll_sims", np.float16), ("attn_code_tokens", np.float32),
					("lr_paths_types", np.int32), ("attn_ast_leaves", np.float32), ("attn_dfg_edges", np.float32),
					("attn_code_ast", np.float32), ("attn_code_dfg", np.float32)]

		return features + self._get_2d_features()

	def build_attention_matrix(self, row, attn_col, num_targets, attn_col_offset):
		num_code_tokens = len(row['code_tokens'])
		attention_matrix = np.full((num_code_tokens, num_targets), self.attn_bias_ignore, dtype=np.float32)

		for j, code_token_idxs in enumerate(row[attn_col]):
			for i in code_token_idxs:
				attention_matrix[i, j + attn_col_offset] = self.attn_bias_attend # adjust for padding

		return attention_matrix

	def masked_attention(self, row):
		row_len = len(row)
		mask = np.triu(np.ones((row_len, row_len), dtype=np.float32) * self.attn_bias_ignore, k=1)
		mask = mask + np.tril(np.zeros((row_len, row_len), dtype=np.float32))

		return mask

	def truncate_ast_dfg(self, data):
		return data
