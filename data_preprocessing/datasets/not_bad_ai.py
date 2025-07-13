from datasets import load_dataset

from data_preprocessing.tasks.task import Task
from data_preprocessing.datasets.dataset import Dataset


class NotBadAi(Dataset):

	def __init__(self, task: Task, split='train'):
		super().__init__(hf_dataset="notbadai/python_functions_reasoning", dataset_save_dir='not_bad_ai', task=task, split=split)

	def get_data_cols(self):
		return 'prompt', 'answer'

	def clean_function(self, sample):
		answer = sample['answer'].strip()

		if answer.startswith("```python"):
			answer = answer[len("```python"):].lstrip()

		if answer.endswith("```</s>"):
			answer = answer[:-len("```</s>")].rstrip()

		sample['answer'] = answer

		return sample

	def load_dataset(self):
		dataset = load_dataset(self.hf_dataset, split="train")
		filtered = dataset.filter(lambda x: x['answer'].count('def') == 1)
		cleaned = filtered.map(self.clean_function)

		return cleaned
