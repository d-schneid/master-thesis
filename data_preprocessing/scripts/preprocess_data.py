import json
import os

from data_preprocessing.parser import Parser


def preprocess_data(batch, data_handler, num_processed_samples_split, global_stats_list, node_type_to_idx, h5_file):
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
		sample_id = f"sample_{num_processed_samples_split[0]}"
		num_processed_samples_split[0] += 1
		sample_group = h5_file.require_group(sample_id)

		for key, array in sample.items():
			sample_group.create_dataset(name=key, data=array, compression="lzf")

	global_stats_list.append({
		"max_code_token_rel_pos": batch_code_token_rel_pos,
		"max_ast_depth": batch_ast_depth,
	})


def store_global_stats(global_stats_list, node_type_to_idx, dataset, num_samples):
	with open(dataset.num_samples_path, 'w') as f:
		json.dump({
			"num_samples": int(num_samples)
		},
			f
		)
	if dataset.task.task == "pretraining" and dataset.split == "train":
		max_code_token_rel_pos = 0
		max_ast_depth = 0
		for stats in global_stats_list:
			max_code_token_rel_pos = max(max_code_token_rel_pos, stats["max_code_token_rel_pos"])
			max_ast_depth = max(max_ast_depth, stats["max_ast_depth"])

		metadata = {
			"num_ast_node_types": int(len(node_type_to_idx)),
			"max_ast_depth": int(max_ast_depth),
			"max_code_token_rel_pos": int(max_code_token_rel_pos),
		}
		os.makedirs(dataset.metadata_dir_dataset, exist_ok=True)
		with open(dataset.metadata_path, 'w') as f:
			json.dump(metadata, f)
		with open(dataset.node_type_to_idx_path, 'w') as f:
			json.dump(node_type_to_idx, f)
