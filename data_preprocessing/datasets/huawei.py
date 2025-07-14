from datasets import load_dataset

from data_preprocessing.tasks.task import Task
from data_preprocessing.datasets.dataset import Dataset

from datasets import Dataset as HFDataset


class Huawei(Dataset):

	def __init__(self, task: Task, split='train'):
		super().__init__(hf_dataset="huawei-noah/python_text2code", dataset_save_dir='huawei', task=task, split=split)

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
		num_train_samples = 2_100_000
		num_test_samples = 80_000
		num_valid_samples = 80_000
		dataset = load_dataset(self.hf_dataset, split="train").select(range(num_train_samples + num_test_samples, num_train_samples + num_test_samples + num_valid_samples)).filter(lambda x: not x['code'].lstrip().startswith("@"))
		converted_dataset = [self.convert_code_string(sample) for sample in dataset]

		return HFDataset.from_list(converted_dataset)
