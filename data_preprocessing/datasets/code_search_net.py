from datasets import load_dataset

from data_preprocessing.tasks.task import Task
from data_preprocessing.datasets.dataset import Dataset


class CodeSearchNet(Dataset):

	def __init__(self, task: Task, split='train'):
		super().__init__(hf_dataset='code-search-net/code_search_net', dataset_save_dir='code_search_net', task=task, split=split)

	def get_data_cols(self):
		return 'func_documentation_string', 'func_code_string'

	def load_dataset(self):
		return load_dataset(self.hf_dataset, self.lang)[self.split]
