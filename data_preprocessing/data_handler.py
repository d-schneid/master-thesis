import os
import re
import tokenize
import json
from io import StringIO
from joblib import Parallel, delayed

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import pandas as pd

from data_preprocessing.tasks.task import Task
from data_preprocessing.tasks.pretraining import Pretraining
from data_preprocessing.datasets.dataset import Dataset

tqdm.pandas()

START_TOK_ID_DFG = 0
PAD_TOK_ID_DFG = 2


class DataHandler:

	def __init__(self, dataset: Dataset, task: Task=Pretraining(),
				 tokenizer=AutoTokenizer.from_pretrained('bigcode/starcoder2-3b')):
		self.save_dir = dataset.save_dir
		self.dataset = dataset
		self.tokenizer = tokenizer
		self.task = task

	def read_dataset(self, batch):
		return pd.DataFrame(self.dataset.read_dataset(batch), columns=['text', 'code'])

	def get_tokenizer_chars(self):
		tokenizer_chars = []

		for i in range(self.tokenizer.vocab_size):
			token = self.tokenizer.decode(i)
			if len(token) == 1:
				tokenizer_chars.append(token)

		tokenizer_chars = [c for c in tokenizer_chars if c != '�']

		return tokenizer_chars

	def remove_comments_and_docstrings(self, source):
		"""
		Returns 'source' minus comments and docstrings.
		"""
		io_obj = StringIO(source)
		out = ""
		prev_toktype = tokenize.INDENT
		last_lineno = -1
		last_col = 0

		for tok in tokenize.generate_tokens(io_obj.readline):
			token_type = tok[0]
			token_string = tok[1]
			start_line, start_col = tok[2]
			end_line, end_col = tok[3]
			line_text = tok[4]

			if start_line > last_lineno:
				last_col = 0 # start at beginning of new line
			if start_col > last_col:
				out += (" " * (start_col - last_col)) # add space between tokens

			# Remove comments:
			if token_type == tokenize.COMMENT:
				pass
			# This series of conditionals removes docstrings:
			elif token_type == tokenize.STRING:
				if prev_toktype != tokenize.INDENT:
					# This is likely a docstring; double-check we're not inside an operator:
					if prev_toktype != tokenize.NEWLINE:
						if start_col > 0:
							out += token_string
			else:
				out += token_string

			prev_toktype = token_type
			last_col = end_col
			last_lineno = end_line

		temp = []
		for row in out.split('\n'):
			if row.strip() != "":
				temp.append(row)
		code = '\n'.join(temp)

		pos = 0
		docstring_quotes = '"""'
		while pos < len(code):
			try:
				start = code[pos:].index(docstring_quotes) + pos
				end = code[start + len(docstring_quotes):].index(docstring_quotes) + start + len(docstring_quotes)
				code = code[:start] + code[end + len(docstring_quotes):]
				pos = start
			except:
				break

		return re.sub(r"\r\n\s*\r\n", '\n', code)

	def preprocess(self, data):
		failed_count = 0
		rows = []
		tokenizer_chars = self.get_tokenizer_chars()
		pbar = tqdm(data.itertuples())

		for row in pbar:
			code = row.code.strip().replace('▁', '_').replace('\r\n', '\n')  # step 1
			code = ''.join(filter(lambda c: c in tokenizer_chars, code))  # step 2
			try:
				code = self.remove_comments_and_docstrings(code)  # step 3
			except:
				failed_count += 1
				pbar.set_description('failed_count=' + str(failed_count))
				continue

			rows.append([row.text.strip(), code])

		data = pd.DataFrame(rows, columns=['text', 'code'])

		return data

	def get_lr_path(self, leaf):
		path = [leaf]
		while path[-1].parent is not None:
			path.append(path[-1].parent)

		return [node.type for node in path]

	def clean_data(self, data):
		data = data.drop(columns=['ast_leaf_tokens', 'ast_leaf_ranges', 'code_tokens_ranges'])
		data = data[data['dfg_edges'].apply(lambda row: row != [])].reset_index(drop=True)

		return data

	def compute_lr_paths_and_ll_sim(self, lr_paths_types):
		num_ast_leaves = min(len(lr_paths_types), self.task.max_seq_len) # upper bound
		ll_sims = np.eye(num_ast_leaves, dtype=np.float16)

		# optimize indices for upper triangular matrix and <START_AST> and <END_AST> tokens
		for i in range(1, num_ast_leaves - 1):
			for j in range(i + 1, num_ast_leaves - 1):
				sim = self.get_ll_sim(lr_paths_types[i], lr_paths_types[j])
				ll_sims[i, j] = ll_sims[j, i] = sim

		return ll_sims

	def get_ll_sim(self, lr_path1, lr_path2):
		num_common_nodes = 1 # root is always common

		for i in range(2, min(len(lr_path1), len(lr_path2)) + 1):
			if lr_path1[-i] == lr_path2[-i]:
				num_common_nodes += 1
			else:
				continue

		numerator = num_common_nodes * num_common_nodes
		denominator = len(lr_path1) * len(lr_path2)

		return numerator / denominator

	def pad_inner_lists(self, row, pad_value=0):
		max_len = max(len(sublist) for sublist in row)

		return np.array([sublist + [pad_value] * (max_len - len(sublist)) for sublist in row], dtype=np.uint16)

	def add_ast_lr_paths_and_ll_sim(self, data, node_type_to_idx):
		all_node_types = set()
		lr_paths_types = []

		for row in tqdm(data.itertuples()):
			curr_lr_paths_nodes_types = [self.get_lr_path(leaf) for leaf in row.ast_leaves]

			curr_lr_paths_types = [['<START_AST>']] + [path for path in curr_lr_paths_nodes_types] + [['<END_AST>']]
			lr_paths_types.append(curr_lr_paths_types)

			all_node_types.update(set(np.concatenate(lr_paths_types[-1])))

		data.drop(columns=['ast_leaves'], inplace=True)
		data['ll_sims'] = Parallel(n_jobs=-1)(
			delayed(self.compute_lr_paths_and_ll_sim)(curr_lr_paths_types)
			for curr_lr_paths_types in tqdm(lr_paths_types)
		)
		max_node_idx = max(node_type_to_idx.values()) if node_type_to_idx else 0
		if max_node_idx == 0:
			node_type_to_idx['<PAD>'] = max_node_idx
			max_node_idx += 1
			node_type_to_idx['<NOT_SEEN>'] = max_node_idx

		for node_type in all_node_types:
			if self.dataset.split != 'train':
				if node_type not in node_type_to_idx:
					node_type_to_idx[node_type] = node_type_to_idx['<NOT_SEEN>']
			elif node_type not in node_type_to_idx:
				max_node_idx += 1
				node_type_to_idx[node_type] = max_node_idx

		data['lr_paths_types'] = [
			[[node_type_to_idx[node_type] for node_type in lr_path] for lr_path in row]
			for row in lr_paths_types
		]
		data['lr_paths_len'] = data['lr_paths_types'].apply(lambda row: np.array([len(sublist) for sublist in row], dtype=np.uint16))
		data['lr_paths_types'] = data['lr_paths_types'].apply(lambda row: self.pad_inner_lists(row))
		max_ast_depth = max(row.shape[1] for row in data['lr_paths_types'])

		return node_type_to_idx, max_ast_depth

	def map_dfg_node_code_token_idices(self, data):
		""""
		A DFG node/variable can correspond to multiple code tokens due to tokenization.
		This function maps each DFG node/variable to the corresponding code tokens.
		"""
		dfg_node_code_token_idxs = []
		dfg_edges = []

		for row in tqdm(data.itertuples()):
			if len(row.dfg_edges) > 0:
				dfg_nodes = sorted(list(set(np.concatenate([[left] + right for left, right in row.dfg_edges]))))
			else:
				dfg_nodes = []

			dfg_node_to_idx = {k: i for i, k in enumerate(dfg_nodes)}
			# DFG was built with the indices of AST leaves
			# Thus, the index of a DFG node can be used to retrieve the corresponding AST leaf and its code tokens
			if max(dfg_nodes) >= len(row.ast_leaf_code_token_idxs):
				dfg_edges.append([])
				dfg_node_code_token_idxs.append([])
				continue
			dfg_node_code_token_idxs.append([row.ast_leaf_code_token_idxs[i] for i in dfg_nodes])
			dfg_edges.append([(dfg_node_to_idx[left], [dfg_node_to_idx[r] for r in right]) for left, right in row.dfg_edges])

		data['dfg_edges'] = dfg_edges
		data['dfg_node_code_token_idxs'] = dfg_node_code_token_idxs
		data['dfg_node_mask'] = [np.array([START_TOK_ID_DFG] + [1] * len(sublist) + [PAD_TOK_ID_DFG], dtype=np.uint8)
								 for sublist in dfg_node_code_token_idxs]
		data = data[data['dfg_edges'].apply(lambda x: x != [])].reset_index(drop=True)

		return data

	def build_df(self, data, node_type_to_idx):
		data = data.drop(columns=['text', 'code'])
		os.makedirs(self.save_dir, exist_ok=True)

		updated_node_type_to_idx, max_ast_depth = self.add_ast_lr_paths_and_ll_sim(data, node_type_to_idx)
		if self.dataset.split != 'train':
			with open(self.dataset.metadata_path_pretraining, 'r') as f:
				metadata_train = json.load(f)
			max_ast_depth = metadata_train['max_ast_depth']
			data = data[data['lr_paths_len'].apply(lambda lengths: np.max(lengths) <= max_ast_depth)].reset_index(drop=True)

		data = self.map_dfg_node_code_token_idices(data)
		self.add_special_tokens(data)
		data = self.task.filter_max_seq_len(data)
		data = self.task.compute_attention_masks(data)
		data = data.drop(columns=['dfg_edges', 'ast_leaf_code_token_idxs'])
		data['code_tokens_rel_pos_ids'] = Parallel(n_jobs=-1)(
			delayed(self.compute_relative_distances)(pos_str)
			for pos_str in tqdm(data['code_tokens_pos_ids'])
		)
		data['text_tokens_rel_pos_ids'] = Parallel(n_jobs=-1)(
			delayed(self.compute_relative_distances)(pos_str)
			for pos_str in tqdm(data['text_tokens_pos_ids'])
		)
		data = data.drop(columns=['code_tokens_pos_ids', 'text_tokens_pos_ids'])
		max_rel_pos = max([row[0][-1] for row in data['code_tokens_rel_pos_ids']])

		cols = (['code_tokens', 'code_tokens_rel_pos_ids', 'lr_paths_types', 'lr_paths_len', 'll_sims', 'dfg_node_mask',]
				+ self.task.get_cols())
		data = data[cols]

		return updated_node_type_to_idx, max_rel_pos, max_ast_depth, data

	def compute_relative_distances(self, pos_ids, max_distance=127):
		# account for padding distance id of 0 that is added when padded in collating a batch
		# thus, 0 should not be assigned as a relative distance
		dist_matrix = np.abs(pos_ids[:, None] - pos_ids[None, :]) + 1
		dist_matrix = np.minimum(dist_matrix, max_distance)

		return dist_matrix.astype(np.uint8)

	def add_special_tokens(self, data):
		data['code_tokens'] = data['code_tokens'].apply(
			lambda x: np.concatenate(([self.tokenizer.bos_token_id], x, [self.tokenizer.eos_token_id])).astype(np.uint16)
		)
		data['code_tokens_pos_ids'] = data['code_tokens'].apply(lambda x: np.arange(len(x)).astype(np.uint16))
		data['text_tokens'] = data['text_tokens'].apply(
			lambda x: np.concatenate(([self.tokenizer.bos_token_id], x, [self.tokenizer.eos_token_id])).astype(np.uint16)
		)
		data['text_tokens_pos_ids'] = data['text_tokens'].apply(lambda x: np.arange(len(x)).astype(np.uint16))

		# account for BOS token
		data['ast_leaf_code_token_idxs'] = data['ast_leaf_code_token_idxs'].apply(lambda x: [[x + 1 for x in sublist] for sublist in x])
		data['dfg_node_code_token_idxs'] = data['dfg_node_code_token_idxs'].apply(lambda x: [[x + 1 for x in sublist] for sublist in x])

		# account for padding of BOS and EOS tokens for DFG sequence
		data['dfg_edges'] = data['dfg_edges'].apply(lambda row: [(x + 1, [y + 1 for y in ys]) for x, ys in row])
