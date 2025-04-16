from structure_aware_data import StructureAwareDataModule
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.nn.parameter import Parameter
import ast
import pandas as pd
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from transformers import AutoTokenizer


if __name__ == "__main__":
	A = torch.randn(82, 4, 13, 896)
	B = torch.randint(0, 14, (82, 4))

	range_tensor = torch.arange(13).view(1, 1, 13).to(B.device)
	mask = range_tensor < B.unsqueeze(-1)

	maskk = mask[0]
	maskkk = mask[1]

	data_module = StructureAwareDataModule()
	data_module.setup()
	dataloader = data_module.train_dataloader()
	for batch in dataloader:
		print(batch)
		break

	train_data = data_module._train_ds
	print(train_data.__len__())
	for row in train_data:
		roww = row
		print(row)
		break