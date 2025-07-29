from structure_aware_customization.dataset.structure_aware_ct_dataset import StructureAwareCTDataset


class StructureAwareEvalCTDataset(StructureAwareCTDataset):

	def __getitem__(self, idx):
		sample = super().__getitem__(idx)
		start_completion_idx = 0

		sample['loss_mask'].zero_()
		sample['loss_mask'][start_completion_idx] = 1
		sample['text_token_ids'][start_completion_idx + 1 :] = 0

		sample['text_token_rel_pos_ids'][start_completion_idx + 1 :, :] = 0
		sample['text_token_rel_pos_ids'][:, start_completion_idx + 1 :] = 0

		sample['attn_text_tokens'][start_completion_idx + 1 :, :] = self.data_handler.task.attn_bias_ignore
		sample['attn_text_tokens'][:, start_completion_idx + 1 :] = self.data_handler.task.attn_bias_ignore

		sample['attn_code_text_T'][start_completion_idx + 1 :, :] = self.data_handler.task.attn_bias_ignore

		return sample
