import os
import json

from data_preprocessing.data_handler import DataHandler
from data_preprocessing.datasets.yolo import Yolo
from data_preprocessing.tasks.code_completion import CodeCompletion
from data_preprocessing.scripts.preprocess_data import preprocess_data, store_global_stats

import h5py


if __name__ == '__main__':
	num_train_samples = 575_000
	num_test_samples = 50_000
	num_valid_samples = 50_000
	task = CodeCompletion()
	dataset = Yolo(task=task, split='validation')
	data = dataset.load_dataset()
	data = data.select(range(num_train_samples + num_test_samples, num_train_samples + num_test_samples + num_valid_samples))
	data_handler = DataHandler(dataset=dataset, task=task)

	num_processed_samples_split = [0]
	global_stats_list = []
	with open(dataset.node_type_to_idx_path, 'r') as f:
		node_type_to_idx = json.load(f)
	os.makedirs(os.path.dirname(dataset.h5_path), exist_ok=True)
	h5_file = h5py.File(dataset.h5_path, 'a')

	data.map(preprocess_data, batched=True, batch_size=10_000, fn_kwargs={"data_handler": data_handler,
																	    "num_processed_samples_split": num_processed_samples_split,
																	    "global_stats_list": global_stats_list,
																	    "node_type_to_idx": node_type_to_idx,
																	    "h5_file": h5_file})

	store_global_stats(global_stats_list, node_type_to_idx, dataset, num_processed_samples_split[0],
					   dataset.num_samples_path, dataset.metadata_dir_dataset, dataset.metadata_path)

	h5_file.close()
