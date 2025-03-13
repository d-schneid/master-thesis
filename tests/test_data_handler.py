from unittest import TestCase
from preprocessing.data_handler import DataHandler


class TestDataHandler(TestCase):

	def _check_remove_comments_and_docstrings(self, code, expected_cleaned_code):
		data_handler = DataHandler(save_dir='')
		cleaned_code = data_handler.remove_comments_and_docstrings(code)

		self.assertEqual(expected_cleaned_code, cleaned_code)

	def test_simple_docsting_and_comment(self):
		code = '''def hello_world(do_print=True):
    """Prints "Hello, World!" to the console.""" # This is a docstring
    hello = "Hello World!"
    if (do_print): print(hello)'''
		expected_cleaned_code = '''def hello_world(do_print=True):
    hello = "Hello World!"
    if (do_print): print(hello)'''
		self._check_remove_comments_and_docstrings(code, expected_cleaned_code)

	def test_multiline_docstring(self):
		code = '''def hello_world(do_print=True):
    """
    Prints "Hello, World!"
    to the console.
    """
    hello = "Hello World!"
    if (do_print): print(hello)'''
		expected_cleaned_code = '''def hello_world(do_print=True):
    hello = "Hello World!"
    if (do_print): print(hello)'''
		self._check_remove_comments_and_docstrings(code, expected_cleaned_code)

	def test_multiple_comments(self):
		code = '''def hello_world(do_print=True):
    # Should be printed
    # to the console
    hello = "Hello World!" # identifier
    # prints to the console
    if (do_print): print(hello)'''
		expected_cleaned_code = '''def hello_world(do_print=True):
    hello = "Hello World!" 
    if (do_print): print(hello)'''
		self._check_remove_comments_and_docstrings(code, expected_cleaned_code)
