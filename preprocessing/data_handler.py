import re
import tokenize
from io import StringIO

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import pandas as pd
tqdm.pandas()
from datasets import load_dataset


class DataHandler:

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
