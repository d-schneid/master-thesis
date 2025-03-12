from unittest import TestCase
from preprocessing.parser import Parser

class TestDfgParser(TestCase):

	def _test_code_structure(self, code, expected_dfg):
		# ast
		parser = Parser()
		tree = parser.ast_parser.parse(bytes(code, 'utf8'))
		ast_token_nodes = parser.tree_to_token_nodes(tree.root_node)  # leave nodes

		# dfg
		ast_tokens_index = [(node.start_point, node.end_point) for node in ast_token_nodes]
		code_rows = code.split('\n')
		code_tokens = [parser.ast_tok_index_to_code_token(x, code_rows) for x in ast_tokens_index]
		ast_tok_index_to_code_tok = {ast_tok_index: (idx, code_tok) for idx, (ast_tok_index, code_tok) in enumerate(zip(ast_tokens_index, code_tokens))}
		dfg, _ = parser.dfg_parser.parse_dfg_python(tree.root_node, ast_tok_index_to_code_tok, {})

		self.assertEqual(len(expected_dfg), len(dfg))
		for i, expected in enumerate(expected_dfg):
			self.assertEqual(expected, dfg[i])

	def test_simple_dfg1(self):
		code = """def hello_world(do_print=True):
	    hello = "Hello World!"
	    if (do_print): print(hello)"""
		self._test_code_structure(code, [
			('hello_world', 1, 'comesFrom', [], []),
			('do_print', 3, 'comesFrom', ['True'], [5]),
			('hello', 8, 'computedFrom', ['"Hello World!"'], [10]),
			('"Hello World!"', 10, 'comesFrom', [], []),
			('do_print', 13, 'comesFrom', ['do_print'], [3]),
			('print', 16, 'comesFrom', [], []),
			('hello', 18, 'comesFrom', ['hello'], [8])
		])

	def test_simple_dfg2(self):
		code = """def add(a, b):
	    result = a + b
	    return result"""
		self._test_code_structure(code, [
			('add', 1, 'comesFrom', [], []),
			('a', 3, 'comesFrom', [], []),
			('b', 5, 'comesFrom', [], []),
			('result', 8, 'computedFrom', ['a', 'b'], [10, 12]),
			('a', 10, 'comesFrom', ['a'], [3]),
			('b', 12, 'comesFrom', ['b'], [5]),
			('result', 14, 'comesFrom', ['result'], [8])
		])

	def test_for_loop(self):
		code = """def sum_list(lst):
	    total = 0
	    for num in lst:
	        total += num
	    return total"""
		self._test_code_structure(code, [
			('sum_list', 1, 'comesFrom', [], []),
			('lst', 3, 'comesFrom', [], []),
			('total', 6, 'computedFrom', ['0'], [8]),
			('0', 8, 'comesFrom', [], []),
			('num', 10, 'computedFrom', ['lst'], [12]),
			('lst', 12, 'comesFrom', ['lst'], [3]),
			('total', 14, 'computedFrom', ['num'], [16]),
			('num', 16, 'comesFrom', ['num'], [10]),
			('total', 18, 'comesFrom', ['total'], [14])
		])

	def test_while_loop(self):
		code = """def sum_until_n(n):
	        total = 0
	        i = 0
	        while i < n:
	            total += i
	            i += 1
	        return total"""
		self._test_code_structure(code, [
			('sum_until_n', 1, 'comesFrom', [], []),
			('n', 3, 'comesFrom', [], []),
			('total', 6, 'computedFrom', ['0'], [8]),
			('0', 8, 'comesFrom', [], []),
			('i', 9, 'computedFrom', ['0'], [11]),
			('0', 11, 'comesFrom', [], []),
			('i', 13, 'comesFrom', ['i'], [9]),
			('n', 15, 'comesFrom', ['n'], [3]),
			('total', 17, 'computedFrom', ['i'], [19]),
			('i', 19, 'comesFrom', ['i'], [9]),
			('i', 20, 'computedFrom', ['1'], [22]),
			('1', 22, 'comesFrom', [], []),
			('total', 24, 'comesFrom', ['total'], [17])
		])

	def test_dict_comprehension(self):
		code = """def dict_comprehension(dataset):
	        dict_ = {key: dataset.columns[key].fill_value for key in dataset.get_column_names() if dataset.is_masked(key) and dataset.dtype(key).kind != "f"}
	        return dict_"""
		self._test_code_structure(code, [
			('dict_comprehension', 1, 'comesFrom', [], []),
			('dataset', 3, 'comesFrom', [], []),
			('dict_', 6, 'computedFrom',
			 ['key', 'dataset', 'columns', 'key', 'fill_value', 'key', 'dataset', 'get_column_names', 'dataset',
			  'is_masked', 'key', 'dataset', 'dtype', 'key', 'kind', '"f"'],
			 [9, 11, 13, 15, 18, 20, 22, 24, 28, 30, 32, 35, 37, 39, 42, 44]),
			('key', 9, 'comesFrom', ['key'], [20]),
			('dataset', 11, 'comesFrom', ['dataset'], [3]),
			('columns', 13, 'comesFrom', [], []),
			('key', 15, 'comesFrom', ['key'], [20]),
			('fill_value', 18, 'comesFrom', [], []),
			('key', 20, 'computedFrom', ['dataset', 'get_column_names'], [22, 24]),
			('dataset', 22, 'comesFrom', ['dataset'], [3]),
			('get_column_names', 24, 'comesFrom', [], []),
			('dataset', 28, 'comesFrom', ['dataset'], [3]),
			('is_masked', 30, 'comesFrom', [], []),
			('key', 32, 'comesFrom', ['key'], [20]),
			('dataset', 35, 'comesFrom', ['dataset'], [3]),
			('dtype', 37, 'comesFrom', [], []),
			('key', 39, 'comesFrom', ['key'], [20]),
			('kind', 42, 'comesFrom', [], []),
			('"f"', 44, 'comesFrom', [], []),
			('dict_', 47, 'comesFrom', ['dict_'], [6]),
		])

	def test_list_comprehension(self):
		code = """def squares(n):
	    return [i * i for i in range(n)]"""
		self._test_code_structure(code, [
			('squares', 1, 'comesFrom', [], []),
			('n', 3, 'comesFrom', [], []),
			('i', 8, 'comesFrom', ['i'], [12]),
			('i', 10, 'comesFrom', ['i'], [12]),
			('i', 12, 'computedFrom', ['range', 'n'], [14, 16]),
			('range', 14, 'comesFrom', [], []),
			('n', 16, 'comesFrom', ['n'], [3])
		])

	def test_lambda(self):
		code = """def lambda(dataset, column_names=None, selection=False):
		    column_names = column_names or dataset.get_column_names(strings=True)
		    result = []
		    for column_name in column_names:
				max_length = dataset[column_name].apply(lambda x: len(x)).max(selection=selection)
				result.append(max_length)
		    return result"""
		self._test_code_structure(code, [
			('lambda', 1, 'comesFrom', [], []),
			('dataset', 3, 'comesFrom', [], []),
			('column_names', 5, 'comesFrom', ['None'], [7]),
			('selection', 9, 'comesFrom', ['False'], [11]),
			('column_names', 14, 'computedFrom',
			 ['column_names', 'dataset', 'get_column_names', 'strings', 'True'],
			 [16, 18, 20, 22, 24]),
			('column_names', 16, 'comesFrom', ['column_names'], [5]),
			('dataset', 18, 'comesFrom', ['dataset'], [3]),
			('get_column_names', 20, 'comesFrom', [], []),
			('strings', 22, 'computedFrom', ['True'], [24]),
			('result', 26, 'computedFrom', [], []),
			('column_name', 31, 'computedFrom', ['column_names'], [33]),
			('column_names', 33, 'comesFrom', ['column_names'], [14]),
			('max_length', 35, 'computedFrom',
			 ['dataset', 'column_name', 'apply', 'x', 'len', 'x', 'max', 'selection', 'selection'],
			 [37, 39, 42, 45, 47, 49, 53, 55, 57]),
			('dataset', 37, 'comesFrom', ['dataset'], [3]),
			('column_name', 39, 'comesFrom', ['column_name'], [31]),
			('apply', 42, 'comesFrom', [], []),
			('x', 45, 'comesFrom', [], []),
			('len', 47, 'comesFrom', [], []),
			('x', 49, 'comesFrom', ['x'], [45]),
			('max', 53, 'comesFrom', [], []),
			('selection', 55, 'computedFrom', ['selection'], [57]),
			('selection', 57, 'comesFrom', ['selection'], [9]),
			('result', 59, 'comesFrom', ['result'], [26]),
			('append', 61, 'comesFrom', [], []),
			('max_length', 63, 'comesFrom', ['max_length'], [35]),
			('result', 66, 'comesFrom', ['result'], [26]),
		])

	def test_if_else(self):
		code = """def if_else(dataset, column_names):
			N = 5
		    for column_name in column_names:
        		if column_name in dataset.get_column_names(strings=True):
            		column = dataset.columns[column_name]
            		shape = (N,) + column.shape[1:]
            		dtype = column.dtype
        		else:
            		dtype = np.float64().dtype
            		shape = (N,)"""
		self._test_code_structure(code, [
			('if_else', 1, 'comesFrom', [], []),
			('dataset', 3, 'comesFrom', [], []),
			('column_names', 5, 'comesFrom', [], []),
			('N', 8, 'computedFrom', ['5'], [10]),
			('5', 10, 'comesFrom', [], []),
			('column_name', 12, 'computedFrom', ['column_names'], [14]),
			('column_names', 14, 'comesFrom', ['column_names'], [5]),
			('column_name', 17, 'comesFrom', ['column_name'], [12]),
			('dataset', 19, 'comesFrom', ['dataset'], [3]),
			('get_column_names', 21, 'comesFrom', [], []),
			('strings', 23, 'computedFrom', ['True'], [25]),
			('column', 28, 'computedFrom', ['dataset', 'columns', 'column_name'], [30, 32, 34]),
			('dataset', 30, 'comesFrom', ['dataset'], [3]),
			('columns', 32, 'comesFrom', [], []),
			('column_name', 34, 'comesFrom', ['column_name'], [12]),
			('shape', 36, 'computedFrom', ['N', 'column', 'shape', '1'], [39, 43, 45, 47]),
			('N', 39, 'comesFrom', ['N'], [8]),
			('column', 43, 'comesFrom', ['column'], [28]),
			('shape', 45, 'comesFrom', [], []),
			('1', 47, 'comesFrom', [], []),
			('dtype', 50, 'computedFrom', ['column', 'dtype'], [52, 54]),
			('column', 52, 'comesFrom', ['column'], [28]),
			('dtype', 54, 'comesFrom', [], []),
			('dtype', 57, 'computedFrom', ['np', 'float64', 'dtype'], [59, 61, 65]),
			('np', 59, 'comesFrom', [], []),
			('float64', 61, 'comesFrom', [], []),
			('dtype', 65, 'comesFrom', [], []),
			('shape', 66, 'computedFrom', ['N'], [69]),
			('N', 69, 'comesFrom', ['N'], [8])
		])
