from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
import tree_sitter_python as tspython
from tree_sitter import Language, Parser as TSParser
from data_preprocessing.dfg_parser import DfgParser


class Parser:

	def __init__(self, tokenizer=AutoTokenizer.from_pretrained('bigcode/starcoder2-7b'), parse_lang=Language(tspython.language())):
		self.tokenizer = tokenizer
		self.ast_parser = TSParser(parse_lang)
		self.dfg_parser = DfgParser()

	def tree_to_token_nodes(self, root_node):
		"""Go down each path in the AST to find leaves"""
		# leave node
		if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
			return [root_node]
		else:
			code_tokens = []
			for child in root_node.children:
				code_tokens += self.tree_to_token_nodes(child)
			return code_tokens

	def ast_tok_index_to_code_token(self, ast_tok_index, code_rows):
		ast_tok_start_point = ast_tok_index[0]
		ast_tok_end_point = ast_tok_index[1]

		row_idx, col_idx = 0, 1
		if ast_tok_start_point[row_idx] == ast_tok_end_point[row_idx]:
			s = code_rows[ast_tok_start_point[row_idx]][ast_tok_start_point[col_idx]:ast_tok_end_point[col_idx]]
		else:
			s = ""
			s += code_rows[ast_tok_start_point[row_idx]][ast_tok_start_point[col_idx]:]
			for i in range(ast_tok_start_point[row_idx] + 1, ast_tok_end_point[row_idx]):
				s += code_rows[i]
			s += code_rows[ast_tok_end_point[row_idx]][:ast_tok_end_point[col_idx]]

		return s

	def extract_structure(self, code):
		# ast
		tree = self.ast_parser.parse(bytes(code, 'utf8'))
		ast_token_nodes = self.tree_to_token_nodes(tree.root_node)  # leave nodes

		# dfg
		ast_tokens_index = [(node.start_point, node.end_point) for node in ast_token_nodes]
		code_rows = code.split('\n')
		code_tokens = [self.ast_tok_index_to_code_token(ast_tok_index, code_rows) for ast_tok_index in ast_tokens_index]
		ast_tok_index_to_code_tok = {ast_tok_index: (idx, code_tok) for idx, (ast_tok_index, code_tok) in enumerate(zip(ast_tokens_index, code_tokens))}
		try:
			dfg, _ = self.dfg_parser.parse_dfg_python(tree.root_node, ast_tok_index_to_code_tok, {})
		except:
			dfg = []

		for d in dfg:
			assert (d[2] == 'comesFrom' or d[2] == 'computedFrom')
		dfg = [(d[1], d[4]) for d in dfg if (len(d[4]) > 0)]  # left comes from right

		return code_tokens, ast_token_nodes, dfg

	def flatten_ast_leaf_ranges(self, code, ast_leaves):
		line_lens = [len(line) + 1 for line in code.split('\n')]
		line_starts = [0] + list(np.cumsum(line_lens))
		row_idx, col_idx = 0, 1

		return [(line_starts[ast_leaf.start_point[row_idx]] + ast_leaf.start_point[col_idx],
				 line_starts[ast_leaf.end_point[row_idx]] + ast_leaf.end_point[col_idx]) for ast_leaf in ast_leaves]

	def tokenize_codes_texts(self, texts, batch_size=1024):
		tokenized_texts = []
		for start in tqdm(range(0, len(texts), batch_size)):
			batch_input_ids = self.tokenizer(texts[start:start + batch_size]).input_ids
			tokenized_texts.extend([np.array(ids, dtype=np.uint16) for ids in batch_input_ids])

		return tokenized_texts

	def add_structure(self, data):
		ast_leaf_tokens, ast_leaves, ast_leaf_ranges, dfg_edges = [], [], [], []
		for row in tqdm(data.itertuples()):
			curr_code_tokens, curr_ast_leaves, curr_dfg_edges = self.extract_structure(row.code)
			ast_leaf_tokens.append(curr_code_tokens)
			ast_leaves.append(curr_ast_leaves)
			ast_leaf_ranges.append(self.flatten_ast_leaf_ranges(row.code, curr_ast_leaves))
			dfg_edges.append(curr_dfg_edges)

		data['ast_leaves'] = ast_leaves  # list of leaf nodes
		data['dfg_edges'] = dfg_edges  # list of "left leaf node index comes from right leaf nodes indices"
		data['ast_leaf_tokens'] = ast_leaf_tokens  # list of code substrings corresponding to each leaf
		data['ast_leaf_ranges'] = ast_leaf_ranges  # list of (start,end) in code for each leaf node

	def add_code_tokens_ranges(self, data):
		pbar = tqdm(data.itertuples())
		ranges = []

		for row in pbar:
			decoded_code_tokens = [self.tokenizer.decode(ct) for ct in row.code_tokens] # [1:-1] 1:-1 to remove <s> and </s>
			decoded_code = ''.join(decoded_code_tokens)  # misses some spaces that are in row.code
			code = row.code

			# map each position in decoded_code to a position in code
			decoded_code_to_code = []
			j = 0
			for i in range(len(decoded_code)):
				if decoded_code[i] == code[j]:
					decoded_code_to_code.append(j)
					j += 1
				elif decoded_code[i] == code[j + 1]:  # if decoded_code missed a space
					decoded_code_to_code.append(j + 1)
					j += 2
				else:
					decoded_code_tokens = []
					break

			# map each code token to a range in code
			decoded_code_idx = 0
			curr_ranges = []
			for ct in decoded_code_tokens:
				s, e = decoded_code_idx, decoded_code_idx + len(ct)
				decoded_code_idx = e
				slice_ = decoded_code_to_code[s:e]
				if slice_:
					curr_ranges.append((min(slice_), 1 + max(slice_)))
				else:
					curr_ranges = []
					break
			ranges.append(curr_ranges)  # first [None] and last [None] for <s> and </s>

		data['code_tokens_ranges'] = ranges
		data = data[data['code_tokens_ranges'].apply(lambda x: x != [])].reset_index(drop=True)

		return data

	def overlap(self, s1, e1, s2, e2):
		return s1 <= s2 < e1 or s2 <= s1 < e2

	def map_ast_leaf_code_token_indices(self, data):
		""""
		An AST leaf can correspond to multiple code tokens due to tokenization.
		This function maps each AST leaf to the corresponding code tokens.
		"""
		ast_leaf_code_token_idxs = []
		for row in tqdm(data.itertuples()):
			curr_code_token_idx = 0
			ast_leaf_code_token_idxs.append([])
			code_tokens_last_idx = len(row.code_tokens) - 1
			for s, e in row.ast_leaf_ranges:
				if s == e:  # there are leaves with start_point=end_point
					ast_leaf_code_token_idxs[-1].append([])
					continue
				# search next code token that corresponds to the current AST leaf
				while not (self.overlap(s, e, row.code_tokens_ranges[curr_code_token_idx][0], row.code_tokens_ranges[curr_code_token_idx][1])):
					curr_code_token_idx += 1
					if curr_code_token_idx > code_tokens_last_idx:
						break

				if curr_code_token_idx > code_tokens_last_idx: break

				overlapping_code_token_idx = curr_code_token_idx
				curr_leaf_token_idxs = []
				# collect all code tokens that correspond to the current AST leaf
				while self.overlap(s, e, row.code_tokens_ranges[overlapping_code_token_idx][0], row.code_tokens_ranges[overlapping_code_token_idx][1]):
					curr_leaf_token_idxs.append(overlapping_code_token_idx)
					overlapping_code_token_idx += 1
					if overlapping_code_token_idx > code_tokens_last_idx: break
				ast_leaf_code_token_idxs[-1].append(curr_leaf_token_idxs)

		data['ast_leaf_code_token_idxs'] = ast_leaf_code_token_idxs
