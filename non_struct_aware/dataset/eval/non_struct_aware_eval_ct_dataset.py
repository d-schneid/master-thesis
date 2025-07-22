from non_struct_aware.dataset.non_struct_aware_ct_dataset import NonStructAwareCTDataset

import torch


class NonStructAwareEvalCTDataset(NonStructAwareCTDataset):

	def __getitem__(self, idx):
		sample = super().__getitem__(idx)
		start_completion_idx = 0

		sample['loss_mask'].zero_()
		sample['loss_mask'][start_completion_idx] = 1
		sample['text_token_ids'][start_completion_idx + 1 :] = 0

		sample['text_token_rel_pos_ids'][start_completion_idx + 1 :, :] = 0
		sample['text_token_rel_pos_ids'][:, start_completion_idx + 1 :] = 0

		sample['attn_text_tokens'][start_completion_idx + 1 :, :] = -1e9
		sample['attn_text_tokens'][:, start_completion_idx + 1 :] = -1e9

		return sample

	def build_attn_mask(self, batch_dict):
		# padded attention mask
		attn_code_tokens = batch_dict['attn_code_tokens']
		attn_text_tokens = batch_dict['attn_text_tokens']
		attn_code_text = batch_dict['attn_code_text']

		# Compute transpose
		attn_code_text_T = attn_code_text.transpose(1, 2).clone()
		attn_code_text_T[:, 0, :] = 0

		# Build block matrices column-wise
		first_col_matrix = torch.cat((attn_code_tokens, attn_code_text_T), dim=1)
		second_col_matrix = torch.cat((attn_code_text, attn_text_tokens), dim=1)

		attn_bias = torch.cat((first_col_matrix, second_col_matrix), dim=2)
		batch_dict['attention_bias'] = attn_bias.unsqueeze(1).bfloat16() # broadcast across all attention heads

		keys_to_remove = self.get_attn_keys()
		for key in keys_to_remove:
			del batch_dict[key]
