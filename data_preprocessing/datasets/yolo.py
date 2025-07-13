from datasets import load_dataset

from data_preprocessing.datasets.dataset import Dataset
from data_preprocessing.tasks.task import Task


class Yolo(Dataset):

	def __init__(self, task: Task, split='train'):
		super().__init__(hf_dataset="YoLo2000/python-stack-functions-filtered", dataset_save_dir='yolo', task=task, split=split)


	def get_data_cols(self):
		return 'sha1', 'content'

	def load_dataset(self):
		yolo_ds = load_dataset(self.hf_dataset, split="train")
		bigcode_ds = load_dataset("bigcode/python-stack-v1-functions-filtered", split="train")

		sha1_set_bigcode_ds = set(bigcode_ds['sha1'])
		filtered_yolo_ds = yolo_ds.filter(lambda sample: sample['sha1'] not in sha1_set_bigcode_ds)

		return filtered_yolo_ds
