import numpy as np
import pandas as pd

from data_preprocessing.tasks.pretraining import Pretraining


class CodeCompletion(Pretraining):

	def __init__(self):
		super().__init__(task='code_completion')

	def bounded_poisson(self, lambda_, low, high):
		while True:
			drawn_sample = np.random.poisson(lambda_)
			if low <= drawn_sample <= high:
				return drawn_sample

	def truncate_dfg_edges(self, dfg_truncate_idx, edges):
		truncated_dfg_edges = []
		for to_node, from_nodes in edges:
			if to_node >= dfg_truncate_idx:
				continue
			updated_from_nodes = [from_node for from_node in from_nodes if from_node < dfg_truncate_idx]
			if updated_from_nodes:
				truncated_dfg_edges.append((to_node, updated_from_nodes))

		return truncated_dfg_edges

	def truncate_ast_dfg_code_idxs_mapping(self, truncate_idx, ast_dfg_code_idxs_mapping):
		truncated_mapping = []
		for idx, sublist in enumerate(ast_dfg_code_idxs_mapping):
			filtered = [x for x in sublist if x <= truncate_idx]
			if not filtered:
				return truncated_mapping, idx
			truncated_mapping.append(filtered)

		return truncated_mapping, len(ast_dfg_code_idxs_mapping)

	def longest_pruned_lr_path(self, row):
		nonzeros = np.nonzero(row)[0]

		return nonzeros[-1] + 1 if nonzeros.size > 0 else 0

	def truncate_sample(self, row):
		code_len = len(row['code_tokens'])
		lambda_ = code_len * 0.5
		truncate_idx = self.bounded_poisson(lambda_, 2, code_len - 2)

		truncated_ast_leaf_code_token_idxs, ast_truncate_idx = self.truncate_ast_dfg_code_idxs_mapping(truncate_idx, row["ast_leaf_code_token_idxs"])

		truncated_ll_sims_row_idxs = np.r_[0:1, 1:ast_truncate_idx + 1, -1]
		truncated_ll_sims_col_idxs = np.r_[0:1, 1:ast_truncate_idx + 1, -1]
		truncated_ll_sims = row["ll_sims"][truncated_ll_sims_row_idxs[:, None], truncated_ll_sims_col_idxs]

		truncated_lr_paths_types = row["lr_paths_types"][np.r_[0:1, 1:ast_truncate_idx + 1, -1]]
		longest_pruned_lr_path = max(self.longest_pruned_lr_path(row) for row in truncated_lr_paths_types)
		truncated_lr_paths_types = np.array([row[:longest_pruned_lr_path] for row in truncated_lr_paths_types])

		truncated_lr_paths_len = row["lr_paths_len"][np.r_[0:1, 1:ast_truncate_idx + 1, -1]]

		truncated_dfg_node_code_token_idxs, dfg_truncate_idx = self.truncate_ast_dfg_code_idxs_mapping(truncate_idx, row["dfg_node_code_token_idxs"])
		truncated_dfg_edges = self.truncate_dfg_edges(dfg_truncate_idx, row["dfg_edges"])
		truncated_dfg_node_mask = np.concatenate([
			row["dfg_node_mask"][0:1], np.ones(dfg_truncate_idx, dtype=row["dfg_node_mask"].dtype), row["dfg_node_mask"][-1:]
		])

		return pd.Series({
			"ast_leaf_code_token_idxs": truncated_ast_leaf_code_token_idxs,
			"ll_sims": truncated_ll_sims,
			"lr_paths_types": truncated_lr_paths_types,
			"lr_paths_len": truncated_lr_paths_len,
			"dfg_node_code_token_idxs": truncated_dfg_node_code_token_idxs,
			"dfg_edges": truncated_dfg_edges,
			"dfg_node_mask": truncated_dfg_node_mask
		})

	def truncate_ast_dfg(self, data):
		data[["ast_leaf_code_token_idxs", "ll_sims", "lr_paths_types", "lr_paths_len", "dfg_node_code_token_idxs",
			  "dfg_edges", "dfg_node_mask"]] = data.apply(self.truncate_sample, axis=1)

		return data
