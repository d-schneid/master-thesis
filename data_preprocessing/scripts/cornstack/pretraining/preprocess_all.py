import os
import json

from data_preprocessing.data_handler import DataHandler
from data_preprocessing.datasets.cornstack import CornStack
from data_preprocessing.tasks.pretraining import Pretraining
from data_preprocessing.scripts.preprocess_data import preprocess_data, store_global_stats

from datasets import Dataset
import h5py


if __name__ == '__main__':
	task = Pretraining()
	split = "train_2"
	dataset = CornStack(task=task, split="train")
	data_handler = DataHandler(dataset=dataset, task=task)

	global_stats_list = []
	os.makedirs(os.path.dirname(dataset.h5_path_2), exist_ok=True)
	h5_file = h5py.File(dataset.h5_path_2, 'a')
	with open(dataset.node_type_to_idx_path, 'r') as f:
		node_type_to_idx = json.load(f)

	batch_size = 10_000
	dataset_size = 50_000 # needs to be <= num_samples_split
	num_samples_split = 3_150_000
	num_processed_samples_split = [0]

	dataset_buffer_lst = []
	streamed_data = dataset.load_dataset()
	streamed_data = streamed_data.select_columns(dataset.get_data_cols())
	shuffled_stream = streamed_data.skip(3_000_000).shuffle(seed=42, buffer_size=100_000)

	for sample in shuffled_stream:
		dataset_buffer_lst.append(sample)

		if len(dataset_buffer_lst) == dataset_size:
			dataset_buffer = Dataset.from_list(dataset_buffer_lst)
			dataset_buffer.map(preprocess_data, batched=True, batch_size=batch_size, fn_kwargs={"data_handler": data_handler,
																		 "num_processed_samples_split": num_processed_samples_split,
																		 "global_stats_list": global_stats_list,
																		 "node_type_to_idx": node_type_to_idx,
																		 "h5_file": h5_file})
		else: continue

		dataset_buffer_lst = []
		missing_samples_split = num_samples_split - num_processed_samples_split[0]
		print(f"Missing samples: {missing_samples_split}")
		if missing_samples_split < dataset_size:
			dataset_size = missing_samples_split

		if missing_samples_split <= 2_100_000 and split == "train_2":
			store_global_stats(global_stats_list, node_type_to_idx, dataset, num_processed_samples_split[0],
							   dataset.num_samples_path_2, dataset.metadata_dir_dataset_2, dataset.metadata_path_2)
			h5_file.close()

			split = "train_3"
			dataset = CornStack(task=task, split="train")
			data_handler = DataHandler(dataset=dataset, task=task)
			global_stats_list = []
			num_samples_split = missing_samples_split
			num_processed_samples_split[0] = 0

			os.makedirs(os.path.dirname(dataset.h5_path_3), exist_ok=True)
			h5_file = h5py.File(dataset.h5_path_3, 'a')

			with open(dataset.node_type_to_idx_path, 'r') as f:
				node_type_to_idx = json.load(f)

		elif missing_samples_split <= 1_050_000 and split == "train_3":
			store_global_stats(global_stats_list, node_type_to_idx, dataset, num_processed_samples_split[0],
							   dataset.num_samples_path_3, dataset.metadata_dir_dataset_3, dataset.metadata_path_3)
			h5_file.close()

			split = "train_4"
			dataset = CornStack(task=task, split="train")
			data_handler = DataHandler(dataset=dataset, task=task)
			global_stats_list = []
			num_samples_split = missing_samples_split
			num_processed_samples_split[0] = 0

			os.makedirs(os.path.dirname(dataset.h5_path_4), exist_ok=True)
			h5_file = h5py.File(dataset.h5_path_4, 'a')

			with open(dataset.node_type_to_idx_path, 'r') as f:
				node_type_to_idx = json.load(f)

		elif missing_samples_split == 0 and split == "train_4":
			store_global_stats(global_stats_list, node_type_to_idx, dataset, num_processed_samples_split[0],
							   dataset.num_samples_path_4, dataset.metadata_dir_dataset_4, dataset.metadata_path_4)
			h5_file.close()

			split = "validation"
			dataset = CornStack(task=task, split=split)
			data_handler = DataHandler(dataset=dataset, task=task)
			global_stats_list = []
			dataset_size = 50_000 # needs to be <= num_samples_split
			num_samples_split = 60_000
			num_processed_samples_split[0] = 0

			os.makedirs(os.path.dirname(dataset.h5_path_2), exist_ok=True)
			h5_file = h5py.File(dataset.h5_path_2, 'a')

			with open(dataset.node_type_to_idx_path, 'r') as f:
				node_type_to_idx = json.load(f)

		elif missing_samples_split == 0 and split == "validation":
			store_global_stats(global_stats_list, node_type_to_idx, dataset, num_processed_samples_split[0],
							   dataset.num_samples_path_2, dataset.metadata_dir_dataset_2, dataset.metadata_path_2)
			h5_file.close()

			split = "test"
			dataset = CornStack(task=task, split=split)
			data_handler = DataHandler(dataset=dataset, task=task)
			global_stats_list = []
			dataset_size = 50_000 # needs to be <= num_samples_split
			num_samples_split = 60_000
			num_processed_samples_split[0] = 0

			os.makedirs(os.path.dirname(dataset.h5_path_2), exist_ok=True)
			h5_file = h5py.File(dataset.h5_path_2, 'a')

			with open(dataset.node_type_to_idx_path, 'r') as f:
				node_type_to_idx = json.load(f)

		elif missing_samples_split == 0 and split == "test":
			store_global_stats(global_stats_list, node_type_to_idx, dataset, num_processed_samples_split[0],
							   dataset.num_samples_path_2, dataset.metadata_dir_dataset_2, dataset.metadata_path_2)
			h5_file.close()
			break
