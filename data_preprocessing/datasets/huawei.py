import os
import json

from data_preprocessing.tasks.task import Task
from data_preprocessing.datasets.dataset import Dataset

from datasets import Dataset as HFDataset
from datasets import load_dataset


class Huawei(Dataset):

	def __init__(self, task: Task, split='train'):
		super().__init__(hf_dataset="huawei-noah/python_text2code", dataset_save_dir='huawei', task=task, split=split)
		self.h5_path_0 = os.path.join(self.save_dir, 'samples_0.h5')
		self.h5_path_1 = os.path.join(self.save_dir, 'samples_1.h5')

		self.num_samples_path_0 = os.path.join(self.save_dir, 'num_samples_0.json')
		self.num_samples_path_1 = os.path.join(self.save_dir, 'num_samples_1.json')

	def get_h5_metadata(self):
		with open(self.num_samples_path_0, 'r') as f0, open(self.num_samples_path_1, 'r') as f1:
			num_samples_0 = json.load(f0)['num_samples']
			num_samples_1 = json.load(f1)['num_samples']

		return [(self.h5_path_0, num_samples_0), (self.h5_path_1, num_samples_1),]

	def get_data_cols(self):
		return 'docstring', 'code'

	def convert_code_string(self, sample, indent_size=4):
		lines = []
		indent_level = 0
		tokens = sample['code'].split()
		i = 0

		while i < len(tokens):
			token = tokens[i]
			if token == '<NEW_LINE>':
				lines.append('\n' + ' ' * (indent_level * indent_size))
			elif token == '<INDENT>':
				lines.append(' ' * indent_size)
				indent_level += 1
			elif token == '<DEDENT>':
				lines[-1] = lines[-1][:-indent_size]
				indent_level = max(0, indent_level - 1)
			else:
				# Merge tokens into a line, handling spacing
				if not lines or lines[-1].endswith('\n'):
					lines.append(token)
				else:
					lines[-1] += ' ' + token
			i += 1

		sample['code'] = ''.join(lines)

		return sample

	def load_dataset(self):
		num_train_samples = 1_200_000
		num_test_samples = 80_000
		num_valid_samples = 80_000
		dataset = load_dataset(self.hf_dataset, split="train").select(range(num_train_samples)).filter(lambda x: not x['code'].lstrip().startswith("@"))
		converted_dataset = [self.convert_code_string(sample) for sample in dataset]

		return HFDataset.from_list(converted_dataset)
