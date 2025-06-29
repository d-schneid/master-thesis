from datasets import load_dataset

from data_preprocessing.datasets.dataset import Dataset
from data_preprocessing.tasks.task import Task


class Stack(Dataset):

	def __init__(self, task: Task, split='train'):
		super().__init__(hf_dataset='bigcode/python-stack-v1-functions-filtered', dataset_save_dir='stack', task=task, split=split)


	def get_data_cols(self):
		return 'sha1', 'content'

	def load_dataset(self):
		return load_dataset(self.hf_dataset, split="train")
