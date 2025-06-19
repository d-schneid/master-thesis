from abc import ABC, abstractmethod
import os

from tqdm import tqdm


class Dataset(ABC):

	def __init__(self, dataset, task, lang='python', split='train'):
		self.absolute_path = '/Users/i741961/Documents/SAP/Masterthesis/Code/master-thesis/data_preprocessing/data/pretraining'
		self.save_dir = os.path.join(self.absolute_path, dataset, task, split)
		self.metadata_path_train = os.path.join(self.absolute_path, dataset, task, 'train', 'metadata.json')
		self.metadata_path = os.path.join(self.save_dir, 'metadata.json')
		self.node_type_to_idx_path = os.path.join(self.absolute_path, 'node_type_to_idx.json')
		self.h5_path = os.path.join(self.save_dir, 'samples.h5')
		self.dataset = dataset
		self.lang = lang
		self.split = split

	@abstractmethod
	def get_data_cols(self):
		"""
		First column should be the text column, second column should be the code column.
		"""
		pass

	@abstractmethod
	def load_dataset(self):
		pass

	def read_dataset(self, batch):
		rows = []
		col1, col2 = self.get_data_cols()

		batch_size = len(next(iter(batch.values())))
		indices = list(range(batch_size))
		pbar = tqdm(indices)

		for i in pbar:
			sample_col1 = batch[col1][i]
			sample_col2 = batch[col2][i]
			rows.append([sample_col1, sample_col2])

		return rows
