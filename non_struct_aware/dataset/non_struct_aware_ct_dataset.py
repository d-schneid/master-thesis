from structure_aware_customization.dataset.structure_aware_dataset import StructureAwareDataset
from data_preprocessing.datasets.dataset import Dataset

import torch


class NonStructAwareCTDataset(StructureAwareDataset):

	def __init__(self, datasets: list[Dataset]) -> None:
		super().__init__(datasets=datasets)

	def __getitem__(self, idx):
		sample = super().__getitem__(idx)
		sample = {
			'code_token_ids': sample['code_token_ids'],
			'text_token_ids': sample['text_token_ids'],
			'code_token_rel_pos_ids': sample['code_token_rel_pos_ids'],
			'text_token_rel_pos_ids': sample['text_token_rel_pos_ids'],
			'attn_code_tokens': sample['attn_code_tokens'],
			'attn_text_tokens': sample['attn_text_tokens'],
			'attn_code_text': sample['attn_code_text'],
		}

		text_tokens = sample['text_token_ids']
		labels = torch.cat([text_tokens[1:], torch.tensor([self.padding_value], dtype=text_tokens.dtype)])
		loss_mask = torch.cat([torch.ones(len(text_tokens[:-1]), dtype=text_tokens.dtype), torch.tensor([0], dtype=text_tokens.dtype)])

		sample['labels'] = labels
		sample['loss_mask'] = loss_mask

		return sample

	def get_2d_tokens_for_max_seq_len_padding(self):
		return ['attn_text_tokens', 'attn_code_text', 'text_token_rel_pos_ids']

	def get_1d_tokens_for_max_seq_len_padding(self):
		return super().get_1d_tokens_for_max_seq_len_padding() + ['text_token_ids']

	def get_1d_keys(self):
		return ['code_token_ids', 'text_token_ids', 'labels', 'loss_mask']

	def get_attn_keys(self):
		return ['attn_code_tokens', 'attn_text_tokens', 'attn_code_text']

	def get_labels_loss_pad_len(self, batch_dict):
		return batch_dict['code_token_ids'][0].size(0)

	def build_attn_mask(self, batch_dict):
		# padded attention mask
		attn_code_tokens = batch_dict['attn_code_tokens']
		attn_text_tokens = batch_dict['attn_text_tokens']
		attn_code_text = batch_dict['attn_code_text']

		# Compute transpose
		attn_code_text_T_shape = attn_code_text.transpose(1, 2).shape
		attn_code_text_T = torch.full(attn_code_text_T_shape, fill_value=0)

		# Build block matrices column-wise
		first_col_matrix = torch.cat((attn_code_tokens, attn_code_text_T), dim=1)
		second_col_matrix = torch.cat((attn_code_text, attn_text_tokens), dim=1)

		attn_bias = torch.cat((first_col_matrix, second_col_matrix), dim=2)
		batch_dict['attention_bias'] = attn_bias.unsqueeze(1).bfloat16() # broadcast across all attention heads

		keys_to_remove = self.get_attn_keys()
		for key in keys_to_remove:
			del batch_dict[key]

	def build_attn_bias(self, batch_dict, first_col_matrix, second_col_matrix, third_col_matrix):
		pass

	def collate_fn(self, batch):
		# Initialize a dictionary to store the batch data
		batch_dict = {}
		self.pad_within_batch(batch, batch_dict)
		self.pad_batch_to_max_seq_len(batch_dict)
		self.build_attn_mask(batch_dict)

		return batch_dict
