from nemo.core import Dataset
from data_preprocessing.data_handler import DataHandler
import torch
import ast


class StructureAwareDataset(Dataset):

	def __init__(self, data_path):
		self.data = DataHandler(save_dir=data_path).get_concat_stored_data()

		self.data['code_tokens'] = self.data['code_tokens'].apply(lambda x: list(map(int, x.split(','))))
		self.data['text_tokens'] = self.data['text_tokens'].apply(lambda x: list(map(int, x.split(','))))
		self.data['ll_sims'] = self.data['ll_sims'].apply(lambda x: [list(map(float, sublist.split(','))) for sublist in x.split(';')])
		self.data['ast_leaf_code_token_idxs'] = self.data['ast_leaf_code_token_idxs'].apply(lambda x: ast.literal_eval(x))
		self.data['lr_paths_types'] = self.data['lr_paths_types'].apply(lambda x: ast.literal_eval(x))
		self.data['dfg_node_code_token_idxs'] = self.data['dfg_node_code_token_idxs'].apply(lambda x: ast.literal_eval(x))
		self.data['dfg_edges'] = self.data['dfg_edges'].apply(lambda x: ast.literal_eval(x))

	def __getitem__(self, idx):
		code_tokens = torch.tensor(self.data.iloc[idx]['code_tokens'], dtype=torch.long)
		text_tokens = torch.tensor(self.data.iloc[idx]['text_tokens'], dtype=torch.long)

		return (code_tokens, text_tokens, self.data.iloc[idx]['ll_sims'], self.data.iloc[idx]['ast_leaf_code_token_idxs'],
				self.data.iloc[idx]['lr_paths_types'], self.data.iloc[idx]['dfg_node_code_token_idxs'], self.data.iloc[idx]['dfg_edges'])

	def __len__(self):
		return len(self.data)
