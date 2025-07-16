from non_struct_aware.dataset.non_struct_aware_pretraining_dataset import NonStructAwarePretrainingDataset


class NonStructAwareCCDataset(NonStructAwarePretrainingDataset):

	def __getitem__(self, idx):
		sample = super().__getitem__(idx)
		sample['loss_mask'][:sample['start_completion_idx'].item()] = 0
		del sample['start_completion_idx']

		return sample
