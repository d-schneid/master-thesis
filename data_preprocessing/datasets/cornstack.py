import os
import json

from datasets import load_dataset

from data_preprocessing.tasks.task import Task
from data_preprocessing.datasets.dataset import Dataset


class CornStack(Dataset):

	def __init__(self, task: Task, split='train'):
		super().__init__(hf_dataset='nomic-ai/cornstack-python-v1', dataset_save_dir='cornstack', task=task, split=split)
		self.h5_path_0 = os.path.join(self.save_dir, 'samples_0.h5')
		self.h5_path_1 = os.path.join(self.save_dir, 'samples_1.h5')
		self.h5_path_2 = os.path.join(self.save_dir, 'samples_2.h5')
		self.h5_path_3 = os.path.join(self.save_dir, 'samples_3.h5')
		self.h5_path_4 = os.path.join(self.save_dir, 'samples_4.h5')

		self.num_samples_path_0 = os.path.join(self.save_dir, 'num_samples_0.json')
		self.num_samples_path_1 = os.path.join(self.save_dir, 'num_samples_1.json')
		self.num_samples_path_2 = os.path.join(self.save_dir, 'num_samples_2.json')
		self.num_samples_path_3 = os.path.join(self.save_dir, 'num_samples_3.json')
		self.num_samples_path_4 = os.path.join(self.save_dir, 'num_samples_4.json')

	def get_h5_metadata(self):
		with (open(self.num_samples_path_0, 'r') as f0,
			  open(self.num_samples_path_1, 'r') as f1,
			  open(self.num_samples_path_2, 'r') as f2,
			  open(self.num_samples_path_3, 'r') as f3,
			  open(self.num_samples_path_4, 'r') as f4):
			num_samples_0 = json.load(f0)['num_samples']
			num_samples_1 = json.load(f1)['num_samples']
			num_samples_2 = json.load(f2)['num_samples']
			num_samples_3 = json.load(f3)['num_samples']
			num_samples_4 = json.load(f4)['num_samples']

		return [(self.h5_path_0, num_samples_0),
				(self.h5_path_1, num_samples_1),
				(self.h5_path_2, num_samples_2),
				(self.h5_path_3, num_samples_3),
				(self.h5_path_4, num_samples_4)]

	def get_data_cols(self):
		return 'query', 'document'

	def load_dataset(self):
		data_files = [f"shard-{i:05d}.jsonl.gz" for i in range(101, 111)]
		return load_dataset(self.hf_dataset, split="train", data_files=data_files)
