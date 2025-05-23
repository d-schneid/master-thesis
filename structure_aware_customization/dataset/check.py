from structure_aware_customization.dataset.structure_aware_data_module import StructureAwareDataModule
from structure_aware_customization.dataset.structure_aware_cc_dataset import StructureAwareCCDataset
from structure_aware_customization.dataset.structure_aware_ct_dataset import StructureAwareCTDataset


if __name__ == "__main__":
	data_module = StructureAwareDataModule(train_dataset=StructureAwareCTDataset(split='train'),
									validation_dataset=StructureAwareCTDataset(split='validation'),
									test_dataset=StructureAwareCTDataset(split='test'))
	data_module.setup()
	dataloader = data_module.train_dataloader()
	for batch in dataloader:
		for key, value in batch.items():
			print(f"{key}: value = {value} shape = {value.shape}")
		break

	dataloader = data_module.test_dataloader()
	for batch in dataloader:
		for key, value in batch.items():
			print(f"{key}: value = {value} shape = {value.shape}")
		break

	train_data = data_module._train_ds
	for row in train_data:
		print(row)
		break