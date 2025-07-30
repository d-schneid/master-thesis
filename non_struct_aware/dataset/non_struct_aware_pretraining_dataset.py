from structure_aware_customization.dataset.structure_aware_dataset import StructureAwareDataset

import torch


class NonStructAwarePretrainingDataset(StructureAwareDataset):

	def __getitem__(self, idx):
		sample = super().__getitem__(idx)
		sample = {
			'code_token_ids': sample['code_token_ids'],
			'code_token_rel_pos_ids': sample['code_token_rel_pos_ids'],
			'attn_code_tokens': sample['attn_code_tokens'],
			'start_completion_idx': sample['start_completion_idx'],
		}

		code_tokens = sample['code_token_ids']
		labels = torch.cat([code_tokens[1:], torch.tensor([self.padding_value], dtype=code_tokens.dtype)])
		loss_mask = torch.cat([torch.ones(len(code_tokens[:-1]), dtype=code_tokens.dtype), torch.tensor([0], dtype=code_tokens.dtype)])

		sample['labels'] = labels
		sample['loss_mask'] = loss_mask

		return sample

	def collate_fn(self, batch):
		# Initialize a dictionary to store the batch data
		batch_dict = {}
		self.pad_within_batch(batch, batch_dict)
		self.pad_batch_to_max_seq_len(batch_dict)
		self.build_attn_mask(batch_dict)

		return batch_dict

	def build_attn_mask(self, batch_dict):
		# padded attention mask
		attn_code_tokens = batch_dict['attn_code_tokens']
		batch_dict['attention_bias'] = attn_code_tokens.unsqueeze(1).bfloat16() # broadcast across all attention heads
		del batch_dict['attn_code_tokens']

	def get_1d_keys(self):
		return ['code_token_ids', 'labels', 'loss_mask']

	def get_attn_keys(self):
		return ['attn_code_tokens']

	def get_labels_loss_pad_len(self, batch_dict):
		return 0

	def get_2d_tokens_for_max_seq_len_padding(self):
		return ['attn_code_tokens', 'code_token_rel_pos_ids']

	def get_1d_tokens_for_max_seq_len_padding(self):
		return super().get_1d_tokens_for_max_seq_len_padding() + ['code_token_ids']

	def build_attn_bias(self, batch_dict, first_col_matrix, second_col_matrix, third_col_matrix):
		pass
