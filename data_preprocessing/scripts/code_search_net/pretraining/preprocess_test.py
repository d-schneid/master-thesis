import json
import os

from data_preprocessing.data_handler import DataHandler
from data_preprocessing.datasets.code_search_net import CodeSearchNet
from data_preprocessing.tasks.pretraining import Pretraining
from data_preprocessing.scripts.preprocess_data import preprocess_data, store_global_stats

import h5py


if __name__ == '__main__':
	task = Pretraining()
	dataset = CodeSearchNet(task=task, split='test')
	data = dataset.load_dataset()
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

	store_global_stats(global_stats_list, node_type_to_idx, dataset, num_processed_samples_split[0])

	h5_file.close()
