import numpy as np
import pandas as pd
import torch

from data_preprocessing.tasks.pretraining import Pretraining


class CodeCompletion(Pretraining):

	def __init__(self):
		super().__init__(task='code_completion')

	def get_cols(self):
		return super().get_cols() + ['start_completion_idx']

	def _generate_sample(self, row):
		update = {
			"start_completion_idx": row["start_completion_idx"],
		}

		return update

	def _decode_next_token(self, logits, batch_no_labels):
		pred_tok_idx = batch_no_labels["loss_mask"].squeeze(0).nonzero(as_tuple=True)[0].item() # only one 1 in loss mask
		pred_tok_id = logits[0, pred_tok_idx].argmax()
		next_tok_idx = pred_tok_idx + 1

		# update code tokens and loss mask for next iteration
		update_code_tok_idx = self.max_seq_len - pred_tok_idx
		batch_no_labels["code_token_ids"][0, -(update_code_tok_idx - 1)] = pred_tok_id
		updated_code_tok_idx = batch_no_labels["code_token_ids"].size(1) - (update_code_tok_idx - 1)

		batch_no_labels["loss_mask"].zero_()
		batch_no_labels["loss_mask"][0, pred_tok_idx + 1] = 1

		return pred_tok_id, updated_code_tok_idx, next_tok_idx

	def _update_rel_pos_ids(self, batch_no_labels, updated_code_tok_idx):
		max_rel_pos_updated_code_tok = updated_code_tok_idx + 1  # adjust for zero-pad token for rel pos
		max_rel_pos = 127
		if max_rel_pos_updated_code_tok > max_rel_pos:
			clipped = torch.arange(max_rel_pos, 0, -1, device=batch_no_labels["code_token_rel_pos_ids"].device)
			pad_len = max_rel_pos_updated_code_tok - max_rel_pos
			padding = torch.full((pad_len,), max_rel_pos, device=batch_no_labels["code_token_rel_pos_ids"].device)
			updated_code_tok_rel_pos_ids = torch.cat([padding, clipped])
		else:
			updated_code_tok_rel_pos_ids = torch.arange(max_rel_pos_updated_code_tok, 0, -1, device=batch_no_labels["code_token_rel_pos_ids"].device)

		batch_no_labels["code_token_rel_pos_ids"][0, :max_rel_pos_updated_code_tok, updated_code_tok_idx] = updated_code_tok_rel_pos_ids
		batch_no_labels["code_token_rel_pos_ids"][0, updated_code_tok_idx, :max_rel_pos_updated_code_tok] = updated_code_tok_rel_pos_ids

	def _update_attention_bias(self, batch_no_labels, next_tok_idx):
		num_code_tokens = batch_no_labels["code_token_ids"].shape[1]
		attention_bias = batch_no_labels["attention_bias"]
		attn_code_start_idx = attention_bias.shape[-1] - num_code_tokens

		# only attend to previous code tokens and not AST/DFG tokens
		attention_bias[0, 0, next_tok_idx, attn_code_start_idx:next_tok_idx + 1] = self.attn_bias_attend

	def _reset_floats(self, batch_no_labels, batch):
		batch_no_labels["attention_bias"][batch_no_labels["attention_bias"] > -1] = self.attn_bias_attend
		batch_no_labels["attention_bias"][batch_no_labels["attention_bias"] <= -1] = self.attn_bias_ignore
		batch_no_labels["ll_sims"] = batch["ll_sims"].clone()

	def decode(self, logits, batch_no_labels, batch):
		pred_tok_id, updated_code_tok_idx, next_tok_idx = self._decode_next_token(logits, batch_no_labels)
		self._update_rel_pos_ids(batch_no_labels, updated_code_tok_idx)
		self._update_attention_bias(batch_no_labels, next_tok_idx)
		self._reset_floats(batch_no_labels, batch)

		return batch_no_labels, pred_tok_id

	def bounded_poisson(self, lambda_, low, high):
		while True:
			drawn_sample = np.random.poisson(lambda_)
			if low <= drawn_sample <= high:
				return drawn_sample

	def truncate_dfg_edges(self, dfg_truncate_idx, edges):
		truncated_dfg_edges = []
		for to_node, from_nodes in edges:
			if to_node > dfg_truncate_idx:
				continue
			updated_from_nodes = [from_node for from_node in from_nodes if from_node <= dfg_truncate_idx]
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
		lambda_ = code_len * 0.5 * np.random.uniform(0.8, 1.2)
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
			"dfg_node_mask": truncated_dfg_node_mask,
			"start_completion_idx": np.array([truncate_idx], dtype=np.uint16)
		})

	def truncate_ast_dfg(self, data):
		data[[
			"ast_leaf_code_token_idxs", "ll_sims", "lr_paths_types", "lr_paths_len", "dfg_node_code_token_idxs",
			"dfg_edges", "dfg_node_mask", "start_completion_idx"
		]] = data.apply(self.truncate_sample, axis=1)

		return data

	def generate_adj_matrix(self, edges, num_nodes):
		adj_matrix = np.full((num_nodes, num_nodes), self.attn_bias_ignore, dtype=np.float32)

		for to_node, from_nodes in edges:
			for from_node in from_nodes:
				adj_matrix[to_node, from_node] = self.attn_bias_attend

		return adj_matrix

	def update_attention_masks(self, row):
		start_completion_idx = int(row["start_completion_idx"][0])
		attn_code_tokens = row["attn_code_tokens"].copy()
		attn_code_tokens[:start_completion_idx + 1, :start_completion_idx + 1] = self.attn_bias_attend

		attn_ast_leaves = np.zeros_like(row["attn_ast_leaves"])

		row["attn_code_tokens"] = attn_code_tokens
		row["attn_ast_leaves"] = attn_ast_leaves

		return row

	def compute_attention_masks(self, data):
		# AST, DFG, and code tokens are already truncated, so truncated attention masks will be computed here
		data = super().compute_attention_masks(data)

		return data.apply(self.update_attention_masks, axis=1)
