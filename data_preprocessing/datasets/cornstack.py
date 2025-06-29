from datasets import load_dataset

from data_preprocessing.tasks.task import Task
from data_preprocessing.datasets.dataset import Dataset


class CornStack(Dataset):

	def __init__(self, task: Task, split='train'):
		super().__init__(hf_dataset='nomic-ai/cornstack-python-v1', dataset_save_dir='cornstack', task=task, split=split)

	def get_data_cols(self):
		return 'query', 'document'

	def load_dataset(self):
		return load_dataset(self.hf_dataset, split="train", streaming=True)
