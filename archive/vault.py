from data_preprocessing.tasks.task import Task
from data_preprocessing.datasets.dataset import Dataset

from datasets import load_dataset


class Vault(Dataset):

	def __init__(self, task: Task, split='train'):
		super().__init__(hf_dataset="NamCyan/thevault-docstringstyle", dataset_save_dir='vault', task=task, split=split)

	def get_data_cols(self):
		return 'original_docstring', 'code'

	def load_dataset(self):
		num_train_samples = 430_000
		num_test_samples = 80_000
		num_valid_samples = 80_000
		return (
			load_dataset(self.hf_dataset, split="train").
			shuffle(seed=42).
			filter(lambda x: x["language"] == "Python").
			select(range(10000)).
			filter(lambda x: self.is_valid_docstring(x["original_docstring"]))
		)
