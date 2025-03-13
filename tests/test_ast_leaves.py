from unittest import TestCase
from preprocessing.parser import Parser


class TestAstLeaves(TestCase):

	def _check_code_leaves(self, code, expected_code_leaves):
		parser = Parser()
		code_tokens, ast_leaves, _ = parser.extract_structure(code)
		code_leaves = list(zip(code_tokens, [node.type for node in ast_leaves]))

		self.assertGreaterEqual(len(code_tokens), 1)
		self.assertGreaterEqual(len(ast_leaves), 1)
		self.assertEqual(len(code_tokens), len(ast_leaves))
		self.assertEqual(len(code_leaves), len(expected_code_leaves))
		for i, expected in enumerate(expected_code_leaves):
			self.assertEqual(expected, code_leaves[i])

	def test_simple1(self):
		code = """def hello_world(do_print=True):
					hello = "Hello World!"
					if (do_print): print(hello)"""
		expected_code_leaves = [
			('def', 'def'),
			('hello_world', 'identifier'),
			('(', '('),
			('do_print', 'identifier'),
			('=', '='),
			('True', 'true'),
			(')', ')'),
			(':', ':'),
			('hello', 'identifier'),
			('=', '='),
			('"Hello World!"', 'string'),
			('if', 'if'),
			('(', '('),
			('do_print', 'identifier'),
			(')', ')'),
			(':', ':'),
			('print', 'identifier'),
			('(', '('),
			('hello', 'identifier'),
			(')', ')')
		]
		self._check_code_leaves(code, expected_code_leaves)

	def test_simple2(self):
		code = """def add(a, b):
					result = a + b
					return result"""
		expected_code_leaves = [
			('def', 'def'),
			('add', 'identifier'),
			('(', '('),
			('a', 'identifier'),
			(',', ','),
			('b', 'identifier'),
			(')', ')'),
			(':', ':'),
			('result', 'identifier'),
			('=', '='),
			('a', 'identifier'),
			('+', '+'),
			('b', 'identifier'),
			('return', 'return'),
			('result', 'identifier')
		]
		self._check_code_leaves(code, expected_code_leaves)

	def test_for_loop(self):
		code = """def sum_list(lst):
					total = 0
					for num in lst:
						total += num
					return total"""
		expected_code_leaves = [
			('def', 'def'),
			('sum_list', 'identifier'),
			('(', '('),
			('lst', 'identifier'),
			(')', ')'),
			(':', ':'),
			('total', 'identifier'),
			('=', '='),
			('0', 'integer'),
			('for', 'for'),
			('num', 'identifier'),
			('in', 'in'),
			('lst', 'identifier'),
			(':', ':'),
			('total', 'identifier'),
			('+=', '+='),
			('num', 'identifier'),
			('return', 'return'),
			('total', 'identifier')
		]
		self._check_code_leaves(code, expected_code_leaves)

	def test_while_loop(self):
		code = """def sum_until_n(n):
					total = 0
					i = 0
					while i < n:
						total += i
						i += 1
					return total"""
		expected_code_leaves = [
			('def', 'def'),
			('sum_until_n', 'identifier'),
			('(', '('),
			('n', 'identifier'),
			(')', ')'),
			(':', ':'),
			('total', 'identifier'),
			('=', '='),
			('0', 'integer'),
			('i', 'identifier'),
			('=', '='),
			('0', 'integer'),
			('while', 'while'),
			('i', 'identifier'),
			('<', '<'),
			('n', 'identifier'),
			(':', ':'),
			('total', 'identifier'),
			('+=', '+='),
			('i', 'identifier'),
			('i', 'identifier'),
			('+=', '+='),
			('1', 'integer'),
			('return', 'return'),
			('total', 'identifier')
		]
		self._check_code_leaves(code, expected_code_leaves)

	def test_dict_comprehension(self):
		code = """def dict_comp():
					return {i: i * i for i in range(10)}"""
		expected_code_leaves = [
			('def', 'def'),
			('dict_comp', 'identifier'),
			('(', '('),
			(')', ')'),
			(':', ':'),
			('return', 'return'),
			('{', '{'),
			('i', 'identifier'),
			(':', ':'),
			('i', 'identifier'),
			('*', '*'),
			('i', 'identifier'),
			('for', 'for'),
			('i', 'identifier'),
			('in', 'in'),
			('range', 'identifier'),
			('(', '('),
			('10', 'integer'),
			(')', ')'),
			('}', '}')
		]
		self._check_code_leaves(code, expected_code_leaves)

	def test_list_comprehension(self):
		code = """def list_comp():
					return [i * i for i in range(10)]"""
		expected_code_leaves = [
			('def', 'def'),
			('list_comp', 'identifier'),
			('(', '('),
			(')', ')'),
			(':', ':'),
			('return', 'return'),
			('[', '['),
			('i', 'identifier'),
			('*', '*'),
			('i', 'identifier'),
			('for', 'for'),
			('i', 'identifier'),
			('in', 'in'),
			('range', 'identifier'),
			('(', '('),
			('10', 'integer'),
			(')', ')'),
			(']', ']')
		]
		self._check_code_leaves(code, expected_code_leaves)

	def test_lambda(self):
		code = """def lambda_func():
					return lambda x: x * x"""
		expected_code_leaves = [
			('def', 'def'),
			('lambda_func', 'identifier'),
			('(', '('),
			(')', ')'),
			(':', ':'),
			('return', 'return'),
			('lambda', 'lambda'),
			('x', 'identifier'),
			(':', ':'),
			('x', 'identifier'),
			('*', '*'),
			('x', 'identifier')
		]
		self._check_code_leaves(code, expected_code_leaves)

	def test_if_else(self):
		code = """def if_else(dataset, column_names):
					N = 5
					for column_name in column_names:
						if N > 0:
							print("N is positive")
						else:
							print("N is non-positive")"""
		expected_code_leaves = [
			('def', 'def'),
			('if_else', 'identifier'),
			('(', '('),
			('dataset', 'identifier'),
			(',', ','),
			('column_names', 'identifier'),
			(')', ')'),
			(':', ':'),
			('N', 'identifier'),
			('=', '='),
			('5', 'integer'),
			('for', 'for'),
			('column_name', 'identifier'),
			('in', 'in'),
			('column_names', 'identifier'),
			(':', ':'),
			('if', 'if'),
			('N', 'identifier'),
			('>', '>'),
			('0', 'integer'),
			(':', ':'),
			('print', 'identifier'),
			('(', '('),
			('"N is positive"', 'string'),
			(')', ')'),
			('else', 'else'),
			(':', ':'),
			('print', 'identifier'),
			('(', '('),
			('"N is non-positive"', 'string'),
			(')', ')')
		]
		self._check_code_leaves(code, expected_code_leaves)
