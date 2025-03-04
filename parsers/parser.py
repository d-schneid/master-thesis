from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
tqdm.pandas()
import numpy as np
from transformers import AutoTokenizer
from io import StringIO
import re
import tokenize
import tree_sitter_python as tspython
from tree_sitter import Language, Parser as TSParser
from .dfg import dfg_python


class Parser:

	def __init__(self, dataset='code_search_net', lang='python', tokenizer=AutoTokenizer.from_pretrained('bigcode/starcoder2-7b')):
		self.dataset = dataset
		self.lang = lang
		self.tokenizer = tokenizer

	def read_dataset(self, max_samples_per_split=None):
		np.random.seed(10)
		dataset = load_dataset(self.dataset, self.lang)
		rows = []

		for split in ['train', 'test', 'validation']:
			num_samples_in_split = len(dataset[split])
			indices = list(range(num_samples_in_split))
			if (max_samples_per_split is not None) and (num_samples_in_split > max_samples_per_split):
				indices = list(map(int, np.random.choice(indices, max_samples_per_split, replace=False)))
			pbar = tqdm(indices)
			pbar.set_description('Reading split=' + split)

			for i in pbar:
				sample = dataset[split][i]
				rows.append([sample['func_documentation_string'], sample['func_code_string']])

		return pd.DataFrame(rows, columns=['text', 'code'])

	def get_tokenizer_chars(self):
		tokenizer_chars = []

		for i in range(self.tokenizer.vocab_size):
			token = self.tokenizer.decode(i)
			if len(token) == 1:
				tokenizer_chars.append(token)

		tokenizer_chars = [c for c in tokenizer_chars if c != '�']

		return tokenizer_chars

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
			ltext = tok[4]

			if start_line > last_lineno:
				last_col = 0
			if start_col > last_col:
				out += (" " * (start_col - last_col))
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
		for x in out.split('\n'):
			if x.strip() != "":
				temp.append(x)
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

	def tree_to_token_nodes(self, root_node):
		if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
			return [root_node]
		else:
			code_tokens = []
			for child in root_node.children:
				code_tokens += self.tree_to_token_nodes(child)
			return code_tokens

	def index_to_code_token(self, index, code):
		start_point = index[0]
		end_point = index[1]
		if start_point[0] == end_point[0]:
			s = code[start_point[0]][start_point[1]:end_point[1]]
		else:
			s = ""
			s += code[start_point[0]][start_point[1]:]
			for i in range(start_point[0] + 1, end_point[0]):
				s += code[i]
			s += code[end_point[0]][:end_point[1]]
		return s

	def extract_structure(self, code, parser):
		# ast
		tree = parser[0].parse(bytes(code, 'utf8'))
		ast_token_nodes = self.tree_to_token_nodes(tree.root_node)  # leaves

		# dfg
		tokens_index = [(node.start_point, node.end_point) for node in ast_token_nodes]
		code = code.split('\n')
		code_tokens = [self.index_to_code_token(x, code) for x in tokens_index]
		index_to_code = {index: (idx, code_) for idx, (index, code_) in enumerate(zip(tokens_index, code_tokens))}
		try:
			dfg, _ = parser[1](tree.root_node, index_to_code, {})
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

	def add_structure(self, data, parse_lang=Language(tspython.language())):
		parser = [TSParser(parse_lang), dfg_python]
		ast_leaf_tokens, ast_leaves, ast_leaf_ranges, dfg_edges = [], [], [], []
		for row in tqdm(data.itertuples()):
			curr_code_tokens, curr_ast_leaves, curr_dfg_edges = self.extract_structure(row.code, parser)
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
