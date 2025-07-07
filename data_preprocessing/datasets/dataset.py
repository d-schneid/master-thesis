from abc import ABC, abstractmethod
import os
import json

from tqdm import tqdm

from data_preprocessing.tasks.task import Task


class Dataset(ABC):

	def __init__(self, hf_dataset, dataset_save_dir, task: Task, lang='python', split='train'):
		self.absolute_path = '/shared/home/xxx/gt_data'
		self.save_dir = os.path.join(self.absolute_path, dataset_save_dir, task.task, split)
		self.num_samples_path = os.path.join(self.save_dir, 'num_samples.json')
		self.metadata_dir = os.path.join(self.absolute_path, 'metadata_pretraining')
		self.metadata_path_pretraining = os.path.join(self.metadata_dir, 'metadata.json')
		self.metadata_dir_dataset = os.path.join(self.metadata_dir, dataset_save_dir)
		self.metadata_path = os.path.join(self.metadata_dir_dataset, 'metadata.json')
		self.node_type_to_idx_path = os.path.join(self.metadata_dir, 'node_type_to_idx.json')
		self.h5_path = os.path.join(self.save_dir, 'samples.h5')
		self.hf_dataset = hf_dataset
		self.lang = lang
		self.split = split
		self.task = task

	def get_h5_metadata(self):
		with open(self.num_samples_path, 'r') as f:
			num_samples = json.load(f)['num_samples']

		return [(self.h5_path, num_samples)]

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
