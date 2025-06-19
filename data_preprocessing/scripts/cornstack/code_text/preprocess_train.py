from itertools import count
import os

from data_preprocessing.data_handler import DataHandler
from data_preprocessing.datasets.cornstack import CornStack
from data_preprocessing.tasks.code_text import CodeText
from data_preprocessing.scripts.preprocess_data import preprocess_data, store_global_stats

import h5py


if __name__ == '__main__':
	task = CodeText()
	dataset = CornStack(task=task.task, split="train")
	data = dataset.load_dataset()
	data_handler = DataHandler(dataset=dataset, task=task)

	sample_counter = count()
	global_stats_list = []
	node_type_to_idx = {}
	os.makedirs(os.path.dirname(dataset.h5_path), exist_ok=True)
	h5_file = h5py.File(dataset.h5_path, 'a')

	preprocessed_data = data.map(preprocess_data, batched=True, batch_size=10000, fn_kwargs={"data_handler": data_handler,
																						   "sample_counter": sample_counter,
																						   "global_stats_list": global_stats_list,
																							 "node_type_to_idx": node_type_to_idx,
																							 "h5_file": h5_file})

	store_global_stats(global_stats_list, node_type_to_idx, dataset, next(sample_counter))

	h5_file.close()
