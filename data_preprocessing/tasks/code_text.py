from data_preprocessing.tasks.task import Task

import numpy as np
import torch


class CodeText(Task):

	def __init__(self):
		super().__init__(task='code_text')

	def _decode_next_token(self, logits, batch_no_labels):
		pred_tok_idx = batch_no_labels["loss_mask"].squeeze(0).nonzero(as_tuple=True)[0].item() # only one 1 in loss mask
		pred_tok_id = logits[0, pred_tok_idx].argmax()
		next_tok_idx = pred_tok_idx + 1

		# update code tokens and loss mask for next iteration
		update_code_tok_idx = self.max_seq_len - pred_tok_idx
		batch_no_labels["text_token_ids"][0, -(update_code_tok_idx - 1)] = pred_tok_id
		updated_text_tok_idx = batch_no_labels["text_token_ids"].size(1) - (update_code_tok_idx - 1)

		batch_no_labels["loss_mask"].zero_()
		batch_no_labels["loss_mask"][0, pred_tok_idx + 1] = 1

		return pred_tok_id, updated_text_tok_idx, next_tok_idx

	def _update_rel_pos_ids(self, batch_no_labels, updated_text_tok_idx):
		max_rel_pos_updated_text_tok = updated_text_tok_idx + 1  # adjust for zero-pad token for rel pos
		max_rel_pos = 127
		if max_rel_pos_updated_text_tok > max_rel_pos:
			clipped = torch.arange(max_rel_pos, 0, -1, device=batch_no_labels["text_token_rel_pos_ids"].device)
			pad_len = max_rel_pos_updated_text_tok - max_rel_pos
			padding = torch.full((pad_len,), max_rel_pos, device=batch_no_labels["text_token_rel_pos_ids"].device)
			updated_text_tok_rel_pos_ids = torch.cat([padding, clipped])
		else:
			updated_text_tok_rel_pos_ids = torch.arange(max_rel_pos_updated_text_tok, 0, -1, device=batch_no_labels["text_token_rel_pos_ids"].device)

		batch_no_labels["text_token_rel_pos_ids"][0, :max_rel_pos_updated_text_tok, updated_text_tok_idx] = updated_text_tok_rel_pos_ids
		batch_no_labels["text_token_rel_pos_ids"][0, updated_text_tok_idx, :max_rel_pos_updated_text_tok] = updated_text_tok_rel_pos_ids

	def _update_attention_bias(self, batch_no_labels, next_tok_idx):
		attention_bias = batch_no_labels["attention_bias"]

		# attend to all previous tokens (i.e. AST, DFG, code, text)
		attention_bias[0, 0, next_tok_idx, :next_tok_idx + 1] = self.attn_bias_attend
		# do not attend to future tokens (i.e. text)
		attention_bias[0, 0, next_tok_idx, next_tok_idx + 1:] = self.attn_bias_ignore

	def _reset_floats(self, batch_no_labels, batch):
		batch_no_labels["attention_bias"][batch_no_labels["attention_bias"] > -1] = self.attn_bias_attend
		batch_no_labels["attention_bias"][batch_no_labels["attention_bias"] <= -1] = self.attn_bias_ignore
		batch_no_labels["ll_sims"] = batch["ll_sims"].clone()

	def decode(self, logits, batch_no_labels, batch):
		pred_tok_id, updated_text_tok_idx, next_tok_idx = self._decode_next_token(logits, batch_no_labels)
		self._update_rel_pos_ids(batch_no_labels, updated_text_tok_idx)
		self._update_attention_bias(batch_no_labels, next_tok_idx)
		self._reset_floats(batch_no_labels, batch)

		return batch_no_labels, pred_tok_id

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

	def filter_max_seq_len(self, data):
		data_filtered = data[
			(data['code_tokens'].apply(len) <= 443) &
			(data['lr_paths_len'].apply(len) <= 293) &
			(data['text_tokens'].apply(len) <= 193) &
			(data['dfg_node_mask'].apply(len) <= 93)
			].reset_index(drop=True)

		cols = self.get_max_seq_len_cols()
		length_sums = data_filtered.apply(lambda row: sum(len(row[col]) for col in cols), axis=1)
		data_filtered = data_filtered[length_sums < self.max_seq_len].reset_index(drop=True)

		return data_filtered

	def get_max_seq_len_cols(self):
		return super().get_max_seq_len_cols() + ['text_tokens']

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

	def _get_1d_features(self):
		return [('text_token_ids', np.int32)]

	def _get_2d_features(self):
		return [('text_token_rel_pos_ids', np.int32), ('attn_text_tokens', np.float32), ('attn_code_text', np.float32),
				('attn_ast_text', np.float32), ('attn_dfg_text', np.float32)]

	def generate_adj_matrix(self, edges, num_nodes):
		adj_matrix = np.full((num_nodes, num_nodes), self.attn_bias_ignore, dtype=np.float32)

		for to_node, from_nodes in edges:
			for from_node in from_nodes:
				adj_matrix[to_node, from_node] = self.attn_bias_attend

		return adj_matrix

	def no_attention(self, row, row_len):
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
			lambda row: self.no_attention(
				row=row,
				row_len=len(row['code_tokens'])
			),
			axis=1
		)

		data['attn_ast_text'] = data.apply(
			lambda row: self.no_attention(
				row=row,
				row_len=len(row['lr_paths_len'])
			),
			axis=1
		)

		data['attn_dfg_text'] = data.apply(
			lambda row: self.no_attention(
				row=row,
				row_len=len(row['dfg_node_mask'])
			),
			axis=1
		)

		return data
