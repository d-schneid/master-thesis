from structure_aware_customization.dataset.structure_aware_pretraining_dataset import StructureAwarePretrainingDataset


class StructureAwareCCDataset(StructureAwarePretrainingDataset):

	def __getitem__(self, idx):
		sample = super().__getitem__(idx)
		sample['loss_mask'][:sample['start_completion_idx'].item()] = 0
		del sample['start_completion_idx']

		return sample
