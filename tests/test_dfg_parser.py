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
