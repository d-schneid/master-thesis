import json
import os

from data_preprocessing.data_handler import DataHandler
from data_preprocessing.dataset.code_search_net import CodeSearchNet
from data_preprocessing.dataset.cornstack import CornStack
from data_preprocessing.parser import Parser
from data_preprocessing.attn_masks.code_completion_attn_mask import CodeCompletionAttnMask
from data_preprocessing.attn_masks.code_text_attn_mask import CodeTextAttnMask


if __name__ == '__main__':
	attn_mask_builder = CodeCompletionAttnMask()
	data_dir = os.path.join('../data/pretraining/', attn_mask_builder.save_dir_suffix)
	dataset = CodeSearchNet()

	global_num_node_types = -1
	global_max_code_token_rel_pos = -1
	global_max_ast_depth = -1

	for split in ['train', 'validation', 'test']:
		data_handler = DataHandler(save_dir=os.path.join(data_dir, split), dataset=dataset,
								   attn_mask_builder=attn_mask_builder)
		data = data_handler.read_dataset(split=split, max_samples=10000)
		data = data_handler.preprocess(data)

		parser = Parser()
		parser.add_structure(data)

		data['code_tokens'] = parser.tokenize_codes_texts(list(data['code']))
		data['text_tokens'] = parser.tokenize_codes_texts(list(data['text']))

		data = parser.add_code_tokens_ranges(data)
		parser.map_ast_leaf_code_token_indices(data)

		data = data_handler.convert_tokens_to_strings(data)
		data = data_handler.clean_data(data)
		split_all_node_types, split_max_code_token_rel_pos = data_handler.store_preprocessed_data(data, num_rows_per_file=10000)
		split_max_ast_depth = data_handler.convert_node_types_to_indices(split_all_node_types)

		global_num_node_types = max(global_num_node_types, len(split_all_node_types))
		global_max_code_token_rel_pos = max(global_max_code_token_rel_pos, split_max_code_token_rel_pos)
		global_max_ast_depth = max(global_max_ast_depth, split_max_ast_depth)

	metadata = {
		"num_ast_node_types": int(global_num_node_types),
		"max_ast_depth": int(global_max_ast_depth),
		"max_code_token_rel_pos": int(global_max_code_token_rel_pos),
	}
	with open(os.path.join(data_dir, 'metadata.json'), 'w') as f:
		json.dump(metadata, f)
