from unittest import TestCase
from data_preprocessing.data_handler import DataHandler
from data_preprocessing.parser import Parser
import pandas as pd


class TestAstLrPaths(TestCase):

	def _check_code_leaves(self, code, expected_lr_paths_types):
		data_handler = DataHandler(save_dir='')
		parser = Parser()
		_, ast_leaves, _ = parser.extract_structure(code)
		data = pd.DataFrame()
		data['ast_leaves'] = [ast_leaves]
		all_node_types = data_handler.add_ast_lr_paths_and_ll_sim(data)


		self.assertGreaterEqual(len(all_node_types), 1)
		self.assertGreaterEqual(len(ast_leaves), 1)
		self.assertEqual(len(data.iloc[0].lr_paths_types), len(ast_leaves))
		self.assertEqual(len(ast_leaves), len(expected_lr_paths_types))
		for i, expected in enumerate(expected_lr_paths_types):
			self.assertEqual(expected, data.iloc[0].lr_paths_types[i])

	def test_simple1(self):
		code = """def hello_world(do_print=True):
					hello = "Hello World!"
					if (do_print): print(hello)"""
		expected_lr_paths_types = [
			['def', 'function_definition', 'module'],
			['identifier', 'function_definition', 'module'],
			['(', 'parameters', 'function_definition', 'module'],
			['identifier', 'default_parameter', 'parameters', 'function_definition', 'module'],
			['=', 'default_parameter', 'parameters', 'function_definition', 'module'],
			['true', 'default_parameter', 'parameters', 'function_definition', 'module'],
			[')', 'parameters', 'function_definition', 'module'],
			[':', 'function_definition', 'module'],
			['identifier', 'assignment', 'expression_statement', 'block', 'function_definition', 'module'],
			['=', 'assignment', 'expression_statement', 'block', 'function_definition', 'module'],
			['string', 'assignment', 'expression_statement', 'block', 'function_definition', 'module'],
			['if', 'if_statement', 'block', 'function_definition', 'module'],
			['(', 'parenthesized_expression', 'if_statement', 'block', 'function_definition', 'module'],
			['identifier', 'parenthesized_expression', 'if_statement', 'block', 'function_definition', 'module'],
			[')', 'parenthesized_expression', 'if_statement', 'block', 'function_definition', 'module'],
			[':', 'if_statement', 'block', 'function_definition', 'module'],
			['identifier', 'call', 'expression_statement', 'block', 'if_statement', 'block', 'function_definition', 'module'],
			['(', 'argument_list', 'call', 'expression_statement', 'block', 'if_statement', 'block', 'function_definition', 'module'],
			['identifier', 'argument_list', 'call', 'expression_statement', 'block', 'if_statement', 'block', 'function_definition', 'module'],
			[')', 'argument_list', 'call', 'expression_statement', 'block', 'if_statement', 'block', 'function_definition', 'module']]
		self._check_code_leaves(code, expected_lr_paths_types)

	def test_simple2(self):
		code = """def add(a, b):
					result = a + b
					return result"""
		expected_lr_paths_types = [
			['def', 'function_definition', 'module'],
			['identifier', 'function_definition', 'module'],
			['(', 'parameters', 'function_definition', 'module'],
			['identifier', 'parameters', 'function_definition', 'module'],
			[',', 'parameters', 'function_definition', 'module'],
			['identifier', 'parameters', 'function_definition', 'module'],
			[')', 'parameters', 'function_definition', 'module'],
			[':', 'function_definition', 'module'],
			['identifier', 'assignment', 'expression_statement', 'block', 'function_definition', 'module'],
			['=', 'assignment', 'expression_statement', 'block', 'function_definition', 'module'],
			['identifier', 'binary_operator', 'assignment', 'expression_statement', 'block', 'function_definition', 'module'],
			['+', 'binary_operator', 'assignment', 'expression_statement', 'block', 'function_definition', 'module'],
			['identifier', 'binary_operator', 'assignment', 'expression_statement', 'block', 'function_definition', 'module'],
			['return', 'return_statement', 'block', 'function_definition', 'module'],
			['identifier', 'return_statement', 'block', 'function_definition', 'module']]
		self._check_code_leaves(code, expected_lr_paths_types)

	def test_for_loop(self):
		code = """def sum_list(lst):
					total = 0
					for num in lst:
						total += num
					return total"""
		expected_lr_paths_types = [
			['def', 'function_definition', 'module'],
			['identifier', 'function_definition', 'module'],
			['(', 'parameters', 'function_definition', 'module'],
			['identifier', 'parameters', 'function_definition', 'module'],
			[')', 'parameters', 'function_definition', 'module'],
			[':', 'function_definition', 'module'],
			['identifier', 'assignment', 'expression_statement', 'block', 'function_definition', 'module'],
			['=', 'assignment', 'expression_statement', 'block', 'function_definition', 'module'],
			['integer', 'assignment', 'expression_statement', 'block', 'function_definition', 'module'],
			['for', 'for_statement', 'block', 'function_definition', 'module'],
			['identifier', 'for_statement', 'block', 'function_definition', 'module'],
			['in', 'for_statement', 'block', 'function_definition', 'module'],
			['identifier', 'for_statement', 'block', 'function_definition', 'module'],
			[':', 'for_statement', 'block', 'function_definition', 'module'],
			['identifier', 'augmented_assignment', 'expression_statement', 'block', 'for_statement', 'block', 'function_definition', 'module'],
			['+=', 'augmented_assignment', 'expression_statement', 'block', 'for_statement', 'block', 'function_definition', 'module'],
			['identifier', 'augmented_assignment', 'expression_statement', 'block', 'for_statement', 'block', 'function_definition', 'module'],
			['return', 'return_statement', 'block', 'function_definition', 'module'],
			['identifier', 'return_statement', 'block', 'function_definition', 'module']]
		self._check_code_leaves(code, expected_lr_paths_types)

	def test_while_loop(self):
		code = """def sum_until_n(n):
					total = 0
					i = 0
					while i < n:
						total += i
						i += 1
					return total"""
		expected_lr_paths_types = [
			['def', 'function_definition', 'module'],
			['identifier', 'function_definition', 'module'],
			['(', 'parameters', 'function_definition', 'module'],
			['identifier', 'parameters', 'function_definition', 'module'],
			[')', 'parameters', 'function_definition', 'module'],
			[':', 'function_definition', 'module'],
			['identifier', 'assignment', 'expression_statement', 'block', 'function_definition', 'module'],
			['=', 'assignment', 'expression_statement', 'block', 'function_definition', 'module'],
			['integer', 'assignment', 'expression_statement', 'block', 'function_definition', 'module'],
			['identifier', 'assignment', 'expression_statement', 'block', 'function_definition', 'module'],
			['=', 'assignment', 'expression_statement', 'block', 'function_definition', 'module'],
			['integer', 'assignment', 'expression_statement', 'block', 'function_definition', 'module'],
			['while', 'while_statement', 'block', 'function_definition', 'module'],
			['identifier', 'comparison_operator', 'while_statement', 'block', 'function_definition', 'module'],
			['<', 'comparison_operator', 'while_statement', 'block', 'function_definition', 'module'],
			['identifier', 'comparison_operator', 'while_statement', 'block', 'function_definition', 'module'],
			[':', 'while_statement', 'block', 'function_definition', 'module'],
			['identifier', 'augmented_assignment', 'expression_statement', 'block', 'while_statement', 'block', 'function_definition', 'module'],
			['+=', 'augmented_assignment', 'expression_statement', 'block', 'while_statement', 'block', 'function_definition', 'module'],
			['identifier', 'augmented_assignment', 'expression_statement', 'block', 'while_statement', 'block', 'function_definition', 'module'],
			['identifier', 'augmented_assignment', 'expression_statement', 'block', 'while_statement', 'block', 'function_definition', 'module'],
			['+=', 'augmented_assignment', 'expression_statement', 'block', 'while_statement', 'block', 'function_definition', 'module'],
			['integer', 'augmented_assignment', 'expression_statement', 'block', 'while_statement', 'block', 'function_definition', 'module'],
			['return', 'return_statement', 'block', 'function_definition', 'module'],
			['identifier', 'return_statement', 'block', 'function_definition', 'module']]
		self._check_code_leaves(code, expected_lr_paths_types)

	def test_dict_comprehension(self):
		code = """def dict_comprehension():
					return {i: i * i for i in range(10)}"""
		expected_lr_paths_types = [
			['def', 'function_definition', 'module'],
			['identifier', 'function_definition', 'module'],
			['(', 'parameters', 'function_definition', 'module'],
			[')', 'parameters', 'function_definition', 'module'],
			[':', 'function_definition', 'module'],
			['return', 'return_statement', 'block', 'function_definition', 'module'],
			['{', 'dictionary_comprehension', 'return_statement', 'block', 'function_definition', 'module'],
			['identifier', 'pair', 'dictionary_comprehension', 'return_statement', 'block', 'function_definition', 'module'],
			[':', 'pair', 'dictionary_comprehension', 'return_statement', 'block', 'function_definition', 'module'],
			['identifier', 'binary_operator', 'pair', 'dictionary_comprehension', 'return_statement', 'block', 'function_definition', 'module'],
			['*', 'binary_operator', 'pair', 'dictionary_comprehension', 'return_statement', 'block', 'function_definition', 'module'],
			['identifier', 'binary_operator', 'pair', 'dictionary_comprehension', 'return_statement', 'block', 'function_definition', 'module'],
			['for', 'for_in_clause', 'dictionary_comprehension', 'return_statement', 'block', 'function_definition', 'module'],
			['identifier', 'for_in_clause', 'dictionary_comprehension', 'return_statement', 'block', 'function_definition', 'module'],
			['in', 'for_in_clause', 'dictionary_comprehension', 'return_statement', 'block', 'function_definition', 'module'],
			['identifier', 'call', 'for_in_clause', 'dictionary_comprehension', 'return_statement', 'block', 'function_definition', 'module'],
			['(', 'argument_list', 'call', 'for_in_clause', 'dictionary_comprehension', 'return_statement', 'block', 'function_definition', 'module'],
			['integer', 'argument_list', 'call', 'for_in_clause', 'dictionary_comprehension', 'return_statement', 'block', 'function_definition', 'module'],
			[')', 'argument_list', 'call', 'for_in_clause', 'dictionary_comprehension', 'return_statement', 'block', 'function_definition', 'module'],
			['}', 'dictionary_comprehension', 'return_statement', 'block', 'function_definition', 'module']]
		self._check_code_leaves(code, expected_lr_paths_types)

	def test_list_comprehension(self):
		code = """def list_comprehension():
					return [i * i for i in range(10)]"""
		expected_lr_paths_types = [
			['def', 'function_definition', 'module'],
			['identifier', 'function_definition', 'module'],
			['(', 'parameters', 'function_definition', 'module'],
			[')', 'parameters', 'function_definition', 'module'],
			[':', 'function_definition', 'module'],
			['return', 'return_statement', 'block', 'function_definition', 'module'],
			['[', 'list_comprehension', 'return_statement', 'block', 'function_definition', 'module'],
			['identifier', 'binary_operator', 'list_comprehension', 'return_statement', 'block', 'function_definition', 'module'],
			['*', 'binary_operator', 'list_comprehension', 'return_statement', 'block', 'function_definition', 'module'],
			['identifier', 'binary_operator', 'list_comprehension', 'return_statement', 'block', 'function_definition', 'module'],
			['for', 'for_in_clause', 'list_comprehension', 'return_statement', 'block', 'function_definition', 'module'],
			['identifier', 'for_in_clause', 'list_comprehension', 'return_statement', 'block', 'function_definition', 'module'],
			['in', 'for_in_clause', 'list_comprehension', 'return_statement', 'block', 'function_definition', 'module'],
			['identifier', 'call', 'for_in_clause', 'list_comprehension', 'return_statement', 'block', 'function_definition', 'module'],
			['(', 'argument_list', 'call', 'for_in_clause', 'list_comprehension', 'return_statement', 'block', 'function_definition', 'module'],
			['integer', 'argument_list', 'call', 'for_in_clause', 'list_comprehension', 'return_statement', 'block', 'function_definition', 'module'],
			[')', 'argument_list', 'call', 'for_in_clause', 'list_comprehension', 'return_statement', 'block', 'function_definition', 'module'],
			[']', 'list_comprehension', 'return_statement', 'block', 'function_definition', 'module']]
		self._check_code_leaves(code, expected_lr_paths_types)

	def test_lambda(self):
		code = """def lambda_func():
					return lambda x: x * x"""
		expected_lr_paths_types = [
			['def', 'function_definition', 'module'],
			['identifier', 'function_definition', 'module'],
			['(', 'parameters', 'function_definition', 'module'],
			[')', 'parameters', 'function_definition', 'module'],
			[':', 'function_definition', 'module'],
			['return', 'return_statement', 'block', 'function_definition', 'module'],
			['lambda', 'lambda', 'return_statement', 'block', 'function_definition', 'module'],
			['identifier', 'lambda_parameters', 'lambda', 'return_statement', 'block', 'function_definition', 'module'],
			[':', 'lambda', 'return_statement', 'block', 'function_definition', 'module'],
			['identifier', 'binary_operator', 'lambda', 'return_statement', 'block', 'function_definition', 'module'],
			['*', 'binary_operator', 'lambda', 'return_statement', 'block', 'function_definition', 'module'],
			['identifier', 'binary_operator', 'lambda', 'return_statement', 'block', 'function_definition', 'module']]
		self._check_code_leaves(code, expected_lr_paths_types)

	def test_if_else(self):
		code = """def if_else(dataset, column_names):
					N = 5
					for column_name in column_names:
						if N > 0:
							print("N is positive")
						else:
							print("N is non-positive")"""
		expected_lr_paths_types = [
			['def', 'function_definition', 'module'],
			['identifier', 'function_definition', 'module'],
			['(', 'parameters', 'function_definition', 'module'],
			['identifier', 'parameters', 'function_definition', 'module'],
			[',', 'parameters', 'function_definition', 'module'],
			['identifier', 'parameters', 'function_definition', 'module'],
			[')', 'parameters', 'function_definition', 'module'],
			[':', 'function_definition', 'module'],
			['identifier', 'assignment', 'expression_statement', 'block', 'function_definition', 'module'],
			['=', 'assignment', 'expression_statement', 'block', 'function_definition', 'module'],
			['integer', 'assignment', 'expression_statement', 'block', 'function_definition', 'module'],
			['for', 'for_statement', 'block', 'function_definition', 'module'],
			['identifier', 'for_statement', 'block', 'function_definition', 'module'],
			['in', 'for_statement', 'block', 'function_definition', 'module'],
			['identifier', 'for_statement', 'block', 'function_definition', 'module'],
			[':', 'for_statement', 'block', 'function_definition', 'module'],
			['if', 'if_statement', 'block', 'for_statement', 'block', 'function_definition', 'module'],
			['identifier', 'comparison_operator', 'if_statement', 'block', 'for_statement', 'block', 'function_definition', 'module'],
			['>', 'comparison_operator', 'if_statement', 'block', 'for_statement', 'block', 'function_definition', 'module'],
			['integer', 'comparison_operator', 'if_statement', 'block', 'for_statement', 'block', 'function_definition', 'module'],
			[':', 'if_statement', 'block', 'for_statement', 'block', 'function_definition', 'module'],
			['identifier', 'call', 'expression_statement', 'block', 'if_statement', 'block', 'for_statement', 'block', 'function_definition', 'module'],
			['(', 'argument_list', 'call', 'expression_statement', 'block', 'if_statement', 'block', 'for_statement', 'block', 'function_definition', 'module'],
			['string', 'argument_list', 'call', 'expression_statement', 'block', 'if_statement', 'block', 'for_statement', 'block', 'function_definition', 'module'],
			[')', 'argument_list', 'call', 'expression_statement', 'block', 'if_statement', 'block', 'for_statement', 'block', 'function_definition', 'module'],
			['else', 'else_clause', 'if_statement', 'block', 'for_statement', 'block', 'function_definition', 'module'],
			[':', 'else_clause', 'if_statement', 'block', 'for_statement', 'block', 'function_definition', 'module'],
			['identifier', 'call', 'expression_statement', 'block', 'else_clause', 'if_statement', 'block', 'for_statement', 'block', 'function_definition', 'module'],
			['(', 'argument_list', 'call', 'expression_statement', 'block', 'else_clause', 'if_statement', 'block', 'for_statement', 'block', 'function_definition', 'module'],
			['string', 'argument_list', 'call', 'expression_statement', 'block', 'else_clause', 'if_statement', 'block', 'for_statement', 'block', 'function_definition', 'module'],
			[')', 'argument_list', 'call', 'expression_statement', 'block', 'else_clause', 'if_statement', 'block', 'for_statement', 'block', 'function_definition', 'module']]
		self._check_code_leaves(code, expected_lr_paths_types)
