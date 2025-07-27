from non_struct_aware.dataset.non_struct_aware_pretraining_dataset import NonStructAwarePretrainingDataset

import torch


class NonStructAwareCCDataset(NonStructAwarePretrainingDataset):

	def __getitem__(self, idx):
		sample = super().__getitem__(idx)
		sample['loss_mask'][:sample['start_completion_idx'].item()] = 0
		del sample['start_completion_idx']

		attn_code_tokens = sample['attn_code_tokens']
		upper_mask = torch.triu(torch.ones_like(attn_code_tokens), diagonal=1).bool()
		attn_code_tokens[upper_mask] = self.data_handler.task.attn_bias_ignore

		return sample
