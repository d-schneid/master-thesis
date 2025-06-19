from data_preprocessing.datasets.dataset import Dataset

from datasets import load_dataset


class CodeSearchNet(Dataset):

	def __init__(self, task, split='train'):
		super().__init__(dataset='code-search-net/code_search_net', task=task, split=split)

	def get_data_cols(self):
		return 'func_documentation_string', 'func_code_string'

	def load_dataset(self):
		return load_dataset(self.dataset, self.lang)[self.split]
