from non_struct_aware.dataset.non_struct_aware_pretraining_dataset import NonStructAwarePretrainingDataset


class NonStructAwareEvalCCDataset(NonStructAwarePretrainingDataset):

	def __getitem__(self, idx):
		sample = super().__getitem__(idx)
		start_completion_idx = sample['start_completion_idx'].item()
		del sample['start_completion_idx']

		sample['loss_mask'].zero_()
		sample['loss_mask'][start_completion_idx] = 1
		sample['code_token_ids'][start_completion_idx + 1 :] = 0

		sample['code_token_rel_pos_ids'][start_completion_idx + 1 :, :] = 0
		sample['code_token_rel_pos_ids'][:, start_completion_idx + 1 :] = 0

		sample['attn_code_tokens'][start_completion_idx + 1 :, :] = -1e9
		sample['attn_code_tokens'][:, start_completion_idx + 1 :] = -1e9

		return sample
