import json
import os

from data_preprocessing.data_handler import DataHandler
from data_preprocessing.parser import Parser


if __name__ == '__main__':
	data_dir = '../data/pretraining/'

	data_handler = DataHandler(save_dir=data_dir)
	data = data_handler.read_dataset(max_samples_per_split=50)
	data = data_handler.preprocess(data)

	parser = Parser()
	parser.add_structure(data)

	data['code_tokens'] = parser.tokenize_codes_texts(list(data['code']))
	data['text_tokens'] = parser.tokenize_codes_texts(list(data['text']))

	parser.add_code_tokens_ranges(data)
	parser.map_ast_leaf_code_token_indices(data)

	data = data_handler.convert_tokens_to_strings(data)
	data = data_handler.clean_data(data)
	all_node_types, max_code_token_rel_pos = data_handler.store_preprocessed_data(data, num_rows_per_file=10000)
	max_ast_depth = data_handler.convert_node_types_to_indices(all_node_types)

	metadata = {
		"num_ast_node_types": int(len(all_node_types)),
		"max_ast_depth": int(max_ast_depth),
		"max_code_token_rel_pos": int(max_code_token_rel_pos),
	}
	with open(os.path.join(data_dir, 'metadata.json'), 'w') as f:
		json.dump(metadata, f)
