from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
import tree_sitter_python as tspython
from tree_sitter import Language, Parser as TSParser
from .dfg import dfg_python
from .dfg_parser import DfgParser


class Parser:

	def __init__(self, tokenizer=AutoTokenizer.from_pretrained('bigcode/starcoder2-7b'), parse_lang=Language(tspython.language())):
		self.tokenizer = tokenizer
		self.ast_parser = TSParser(parse_lang)
		self.dfg_parser = DfgParser()

	def tree_to_token_nodes(self, root_node):
		# leave node
		if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
			return [root_node]
		else:
			code_tokens = []
			for child in root_node.children:
				code_tokens += self.tree_to_token_nodes(child)
			return code_tokens

	def ast_tok_index_to_code_token(self, ast_tokens_index, code_rows):
		ast_tok_start_point = ast_tokens_index[0]
		ast_tok_end_point = ast_tokens_index[1]

		row_idx = 0
		col_idx = 1
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
		code_tokens = [self.ast_tok_index_to_code_token(x, code_rows) for x in ast_tokens_index]
		ast_tok_index_to_code_tok = {ast_tok_index: (idx, code_tok) for idx, (ast_tok_index, code_tok) in enumerate(zip(ast_tokens_index, code_tokens))}
		try:
			dfg, _ = self.dfg_parser.parse_dfg_python(tree.root_node, ast_tok_index_to_code_tok, {})
		except:
			dfg = []

		for d in dfg:
			assert (d[2] == 'comesFrom' or d[2] == 'computedFrom')
		dfg = [(d[1], d[4]) for d in dfg if (len(d[4]) > 0)]  # left comes from right

		return code_tokens, ast_token_nodes, dfg

	def format_node_ranges(self, code, nodes):
		line_lens = [len(line) + 1 for line in code.split('\n')]
		line_starts = [0] + list(np.cumsum(line_lens))
		return [(line_starts[node.start_point[0]] + node.start_point[1],
				 line_starts[node.end_point[0]] + node.end_point[1]) for node in nodes]

	def tokenize_codes_texts(self, texts, batch_size=1024):
		tokenized_texts = []
		for start in tqdm(range(0, len(texts), batch_size)):
			tokenized_texts += self.tokenizer(texts[start:start + batch_size]).input_ids

		return tokenized_texts

	def add_structure(self, data):
		ast_leaf_tokens, ast_leaves, ast_leaf_ranges, dfg_edges = [], [], [], []
		for row in tqdm(data.itertuples()):
			curr_code_tokens, curr_ast_leaves, curr_dfg_edges = self.extract_structure(row.code)
			ast_leaf_tokens.append(curr_code_tokens)
			ast_leaves.append(curr_ast_leaves)
			ast_leaf_ranges.append(self.format_node_ranges(row.code, curr_ast_leaves))
			dfg_edges.append(curr_dfg_edges)

		data['ast_leaves'] = ast_leaves  # list of leaf nodes
		data['dfg_edges'] = dfg_edges  # list of "left leaf node index comes from right leaf nodes indices"
		data['ast_leaf_tokens'] = ast_leaf_tokens  # list of code substrings corresponding to each leaf
		data['ast_leaf_ranges'] = ast_leaf_ranges  # list of (start,end) in code for each leaf node

	def get_code_tokens_ranges(self, data):
		pbar = tqdm(data.itertuples())
		ranges = []

		for row in pbar:
			code_tokens = [self.tokenizer.decode(ct) for ct in row.code_tokens] # [1:-1] 1:-1 to remove <s> and </s>
			code2 = ''.join(code_tokens)  # misses some spaces that are in row.code
			code = row.code

			# map each position in code2 to a position in code
			code2_to_code = []
			j = 0
			for i in range(len(code2)):
				if code2[i] == code[j]:
					code2_to_code.append(j)
					j += 1
				elif code2[i] == code[j + 1]:  # if code2 missed a space
					code2_to_code.append(j + 1)
					j += 2
				else:
					raise Exception('Character "' + code2[i] + '" from tokenized code not found in code.')

			# map each code token to a range in code
			code2_idx = 0
			curr_ranges = []
			for ct in code_tokens:
				s, e = code2_idx, code2_idx + len(ct)
				code2_idx = e
				curr_ranges.append((min(code2_to_code[s:e]), 1 + max(code2_to_code[s:e])))
			ranges.append(curr_ranges)  # first [None] and last [None] for <s> and </s>

		data['code_tokens_ranges'] = ranges

	def overlap(self, s1, e1, s2, e2):
		return s1 <= s2 < e1 or s2 <= s1 < e2

	def get_leaf_code_token_indices(self, data):
		ast_leaf_token_idxs = []
		for row in tqdm(data.itertuples()):
			j = 0
			ast_leaf_token_idxs.append([])
			code_tokens_last_idx = len(row.code_tokens) - 1
			for s, e in row.ast_leaf_ranges:
				if s == e:  # there are leaves with start_point=end_point
					ast_leaf_token_idxs[-1].append([])
					continue
				while not (self.overlap(s, e, row.code_tokens_ranges[j][0], row.code_tokens_ranges[j][1])):
					j += 1
				jj = j
				curr_leaf_token_idxs = []
				while self.overlap(s, e, row.code_tokens_ranges[jj][0], row.code_tokens_ranges[jj][1]):
					curr_leaf_token_idxs.append(jj)
					jj += 1
					if jj > code_tokens_last_idx:
						break
				ast_leaf_token_idxs[-1].append(curr_leaf_token_idxs)

		data['ast_leaf_code_token_idxs'] = ast_leaf_token_idxs

	def get_lr_path(self, leaf):
		path = [leaf]
		while path[-1].parent is not None:
			path.append(path[-1].parent)
		return path

	def get_ll_sim(self, p1, p2):
		common = 1
		for i in range(2, min(len(p1), len(p2)) + 1):
			if p1[-i] == p2[-i]:
				common += 1
			else:
				break
		return common * common / (len(p1) * len(p2))

	def get_ast_lr_paths_and_ll_sim(self, data):
		sims = []
		lr_paths = []
		all_node_types = set()
		for i, row in tqdm(enumerate(data.itertuples())):
			L = min(len(row.ast_leaves), 512)
			curr_paths = [self.get_lr_path(leaf) for leaf in row.ast_leaves]
			curr_sims = np.ones((L, L))
			for i in range(L - 1):
				for j in range(i + 1, L):
					curr_sims[i, j] = curr_sims[j, i] = self.get_ll_sim(curr_paths[i], curr_paths[j])
			sims.append(';'.join([','.join(list(map(str, row))) for row in curr_sims]))
			lr_paths.append([[node.type for node in path] for path in curr_paths])
			all_node_types.update(set(np.concatenate(lr_paths[-1])))
		data.drop(columns=['ast_leaves'], inplace=True)
		data['ll_sims'] = sims
		data['lr_paths_types'] = lr_paths

		return all_node_types

	def process_dfg_edges(self, data):
		dfg_node_code_token_idxs = []
		dfg_edges = []
		for row in tqdm(data.itertuples()):
			if len(row.dfg_edges) > 0:
				nodes = sorted(list(set(np.concatenate([[left] + right for left, right in row.dfg_edges]))))
			else:
				nodes = []
			node_to_idx = {k: i for i, k in enumerate(nodes)}
			dfg_node_code_token_idxs.append([row.ast_leaf_code_token_idxs[i] for i in nodes])
			dfg_edges.append([(node_to_idx[left], [node_to_idx[r] for r in right]) for left, right in row.dfg_edges])
		data['dfg_edges'] = dfg_edges
		data['dfg_node_code_token_idxs'] = dfg_node_code_token_idxs

	def parse_list_of_lists(self, s, type_=int):
		list_of_lists = s[1:-2].split('], ')
		if type_ == str:
			list_of_lists = [[t[1:-1].replace('\\n', '\n').replace('\\\\', '\\') for t in x[1:].split(', ')] \
							 for x in list_of_lists]
		elif type_ == int:
			list_of_lists = [[int(t) for t in x[1:].split(', ')] for x in list_of_lists]
		else:
			raise Exception('Unknown value for type_')
		return list_of_lists
