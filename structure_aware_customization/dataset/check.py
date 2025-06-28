from data_preprocessing.datasets.code_search_net import CodeSearchNet
from data_preprocessing.tasks.pretraining import Pretraining
from data_preprocessing.tasks.code_text import CodeText
from structure_aware_customization.dataset.structure_aware_data_module import StructureAwareDataModule
from structure_aware_customization.dataset.structure_aware_pretraining_dataset import StructureAwarePretrainingDataset
from structure_aware_customization.dataset.structure_aware_ct_dataset import StructureAwareCTDataset


if __name__ == "__main__":
	task = Pretraining()
	train_ds = CodeSearchNet(task=task.task, split="train")
	valid_ds = CodeSearchNet(task=task.task, split="validation")
	test_ds = CodeSearchNet(task=task.task, split="test")
	data_module = StructureAwareDataModule(train_dataset=StructureAwarePretrainingDataset(dataset=train_ds),
										   validation_dataset=StructureAwarePretrainingDataset(dataset=valid_ds),
										   test_dataset=StructureAwarePretrainingDataset(dataset=test_ds), )
	data_module.setup()

	dataloader = data_module.train_dataloader()
	for batch in dataloader:
		for key, value in batch.items():
			print(f"{key}: value = {value} shape = {value.shape}")
		break

	dataloader = data_module.val_dataloader()
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
