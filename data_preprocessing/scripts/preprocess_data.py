import json

from data_preprocessing.parser import Parser

import h5py
import numpy as np


def initialize_hdf5_file(h5_path, num_samples_upper_bound, task):
	h5_file = h5py.File(h5_path, "a")
	features_1d = task.get_1d_features()
	features_2d = task.get_2d_features()


	for feature, dtype in features_1d:
		h5_file.create_dataset(
			feature,
			shape=(num_samples_upper_bound,),
			dtype=h5py.special_dtype(vlen=dtype),
			chunks=True,
			compression="lzf"
		)

	for feature, dtype in features_2d:
		h5_file.create_dataset(f"{feature}_data", shape=(0,), maxshape=(None,), dtype=dtype, chunks=True, compression="lzf")
		h5_file.create_dataset(f"{feature}_offsets", shape=(num_samples_upper_bound,), dtype=np.int64, chunks=True, compression="lzf")
		h5_file.create_dataset(f"{feature}_shapes", shape=(num_samples_upper_bound, 2), dtype=np.int32, chunks=True, compression="lzf")

	return h5_file


def append_batch_to_hdf5(batch_samples, h5_file, sample_counter, features_1d, features_2d):
	for sample in batch_samples:
		idx = next(sample_counter)

		for feature, dtype in features_1d:
			ds = h5_file[feature]
			ds[idx] = sample[feature]

		# 2D features: offsets and shapes pre-allocated, data_ds still needs resizing
		for feature, dtype in features_2d:
			array = sample[feature]
			flat = array.ravel().astype(dtype)
			shape = array.shape

			offset_ds = h5_file[f"{feature}_offsets"]
			shape_ds = h5_file[f"{feature}_shapes"]
			data_ds = h5_file[f"{feature}_data"]

			# Append flat data dynamically
			offset = data_ds.shape[0]
			new_len = offset + flat.shape[0]
			data_ds.resize((new_len,))
			data_ds[offset:new_len] = flat

			# Assign offset and shape at idx
			offset_ds[idx] = offset
			shape_ds[idx] = shape


def preprocess_data(batch, data_handler, sample_counter, global_stats_list, node_type_to_idx, h5_file):
	data = data_handler.read_dataset(batch=batch)
	data = data_handler.preprocess(data)

	parser = Parser()
	parser.add_structure(data)

	data['code_tokens'] = parser.tokenize_codes_texts(list(data['code']))
	data['text_tokens'] = parser.tokenize_codes_texts(list(data['text']))

	data = parser.add_code_tokens_ranges(data)
	parser.map_ast_leaf_code_token_indices(data)

	data = data_handler.clean_data(data)
	node_type_to_idx, batch_code_token_rel_pos, batch_ast_depth, data = data_handler.build_df(data, node_type_to_idx)

	batch_samples = []
	for i, row in data.iterrows():
		sample = data_handler.task.generate_sample(row)
		batch_samples.append(sample)

	for sample in batch_samples:
		sample_id = f"sample_{next(sample_counter)}"
		sample_group = h5_file.require_group(sample_id)

		for key, array in sample.items():
			sample_group.create_dataset(name=key, data=array, compression="lzf")

	#append_batch_to_hdf5(batch_samples, h5_file, sample_counter, data_handler.task.get_1d_features(), data_handler.task.get_2d_features())

	global_stats_list.append({
		"max_code_token_rel_pos": batch_code_token_rel_pos,
		"max_ast_depth": batch_ast_depth,
	})


def store_global_stats(global_stats_list, node_type_to_idx, dataset, num_samples):
	if dataset.split != "train":
		with open(dataset.metadata_path, 'w') as f:
			json.dump({
				"num_samples": int(num_samples)
			},
				f)
	else:
		max_code_token_rel_pos = 0
		max_ast_depth = 0
		for stats in global_stats_list:
			max_code_token_rel_pos = max(max_code_token_rel_pos, stats["max_code_token_rel_pos"])
			max_ast_depth = max(max_ast_depth, stats["max_ast_depth"])

		metadata = {
			"num_ast_node_types": int(len(node_type_to_idx)),
			"max_ast_depth": int(max_ast_depth),
			"max_code_token_rel_pos": int(max_code_token_rel_pos),
			"num_samples": int(num_samples),
		}
		with open(dataset.metadata_path, 'w') as f:
			json.dump(metadata, f)
		with open(dataset.node_type_to_idx_path, 'w') as f:
			json.dump(node_type_to_idx, f)
