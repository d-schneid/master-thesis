from data_preprocessing.tasks.task import Task
from data_preprocessing.datasets.dataset import Dataset

from datasets import load_dataset


class Docstring(Dataset):

	def __init__(self, task: Task, split='train'):
		super().__init__(hf_dataset="NamCyan/thevault-docstringstyle", dataset_save_dir='docstring', task=task, split=split)

	def get_data_cols(self):
		return 'original_docstring', 'code'

	def load_dataset(self):
		return (
			load_dataset(self.hf_dataset, split="train").
			filter(lambda x: x["language"] == "Python").
			filter(lambda x: self.is_valid_docstring(x["original_docstring"]))
		)
