from data_preprocessing.datasets.dataset import Dataset

from datasets import load_dataset


class CornStack(Dataset):

	def __init__(self, task, split='train'):
		super().__init__(dataset='nomic-ai/cornstack-python-v1', task=task, split=split)

	def get_data_cols(self):
		return 'query', 'document'

	def load_dataset(self):
		return load_dataset(self.dataset, split=self.split, streaming=True)
