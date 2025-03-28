from data_preprocessing.data_handler import DataHandler
from data_preprocessing.parser import Parser


if __name__ == '__main__':
	data_handler = DataHandler(save_dir='../data/pretraining/')
	data = data_handler.read_dataset(max_samples_per_split=10)
	data = data_handler.preprocess(data)

	parser = Parser()
	parser.add_structure(data)

	data['code_tokens'] = parser.tokenize_codes_texts(list(data['code']))
	data['text_tokens'] = parser.tokenize_codes_texts(list(data['text']))

	parser.add_code_tokens_ranges(data)
	parser.map_ast_leaf_code_token_indices(data)

	data = data_handler.convert_tokens_to_strings(data)
	all_node_types = data_handler.store_preprocessed_data(data, num_rows_per_file=10000)
	data_handler.convert_node_types_to_indices(all_node_types)
	data_handler.reduce_ll_sims()
