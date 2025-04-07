from data_handler import DataHandler
import ast
from torch.utils.data import Dataset, DataLoader
import torch
import lightning.pytorch as pl
from typing import Optional, List, TYPE_CHECKING
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from nemo.utils.import_utils import safe_import
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence

_, HAVE_TE = safe_import("transformer_engine")

if TYPE_CHECKING:
	from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


class StructureAwareDataModule(pl.LightningDataModule):

	def __init__(
			self,
			seq_length: int = 2048,
			tokenizer: Optional["TokenizerSpec"] = None,
			micro_batch_size: int = 4,
			global_batch_size: int = 16,
			rampup_batch_size: Optional[List[int]] = None,
			num_train_samples: int = 10_000,
			num_val_samples: int = 10_000,
			num_test_samples: int = 10_000,
			num_workers: int = 8,
			pin_memory: bool = True,
			persistent_workers: bool = False,
			create_attention_mask: bool = False,
			vocab_file: Optional[str] = None,
			merges_file: Optional[str] = None
	):
		super().__init__()
		self.seq_length = seq_length
		self.micro_batch_size = micro_batch_size
		self.global_batch_size = global_batch_size
		self.num_train_samples = num_train_samples
		self.num_val_samples = num_val_samples
		self.num_test_samples = num_test_samples
		self.num_workers = num_workers
		self.pin_memory = pin_memory
		self.persistent_workers = persistent_workers
		self.create_attention_mask = create_attention_mask or not HAVE_TE

		if tokenizer is None:
			from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

			self.tokenizer = get_nmt_tokenizer(
				"megatron", "GPT2BPETokenizer", vocab_file=vocab_file, merges_file=merges_file
			)
		else:
			self.tokenizer = tokenizer

		self.data_sampler = MegatronDataSampler(
			seq_len=self.seq_length,
			micro_batch_size=self.micro_batch_size,
			global_batch_size=self.global_batch_size,
			rampup_batch_size=rampup_batch_size,
		)

	def setup(self, stage: str = "") -> None:
		self._train_ds = StructureAwareDataset('../data/pretraining')
		self._validation_ds = StructureAwareDataset('../data/pretraining')
		self._test_ds = StructureAwareDataset('../data/pretraining')

	def train_dataloader(self) -> TRAIN_DATALOADERS:
		if not hasattr(self, "_train_ds"):
			self.setup()
		return self._create_dataloader(self._train_ds)

	def val_dataloader(self) -> EVAL_DATALOADERS:
		if not hasattr(self, "_validation_ds"):
			self.setup()
		return self._create_dataloader(self._validation_ds)

	def test_dataloader(self) -> EVAL_DATALOADERS:
		if not hasattr(self, "_test_ds"):
			self.setup()
		return self._create_dataloader(self._test_ds)

	def _create_dataloader(self, dataset, **kwargs) -> DataLoader:
		return DataLoader(
			dataset,
			num_workers=self.num_workers,
			pin_memory=self.pin_memory,
			persistent_workers=self.persistent_workers,
			collate_fn=dataset.collate_fn,
			**kwargs,
		)


class StructureAwareDataset(Dataset):

	def __init__(self, data_path) -> None:
		super().__init__()
		self.data = DataHandler(save_dir=data_path).get_concat_stored_data()

		self.data['code_tokens'] = self.data['code_tokens'].apply(lambda x: list(map(int, x.split(','))))
		self.data['text_tokens'] = self.data['text_tokens'].apply(lambda x: list(map(int, x.split(','))))
		self.data['ll_sims'] = self.data['ll_sims'].apply(lambda x: [list(map(float, sublist.split(','))) for sublist in x.split(';')])
		self.data['ast_leaf_code_token_idxs'] = self.data['ast_leaf_code_token_idxs'].apply(lambda x: ast.literal_eval(x))
		self.data['lr_paths_types'] = self.data['lr_paths_types'].apply(lambda x: ast.literal_eval(x))
		self.data['dfg_node_code_token_idxs'] = self.data['dfg_node_code_token_idxs'].apply(lambda x: ast.literal_eval(x))
		self.data['dfg_edges'] = self.data['dfg_edges'].apply(lambda x: ast.literal_eval(x))

	def __getitem__(self, idx):
		batch = {
			'code_tokens': self.data.iloc[idx]['code_tokens'],
			'text_tokens': self.data.iloc[idx]['text_tokens'],
			#'ll_sims': self.data.iloc[idx]['ll_sims'],
			#'ast_leaf_code_token_idxs': self.data.iloc[idx]['ast_leaf_code_token_idxs'],
			#'lr_paths_types': self.data.iloc[idx]['lr_paths_types'],
			#'dfg_node_code_token_idxs': self.data.iloc[idx]['dfg_node_code_token_idxs'],
			#'dfg_edges': self.data.iloc[idx]['dfg_edges']
		}

		return batch

	def __len__(self) -> int:
		return len(self.data)

	def _collate_fn(self, batch):
		"""
		A default implementation of a collation function.
		Users should override this method to define custom data loaders.
		"""
		return data.dataloader.default_collate(batch)

	def collate_fn(self, batch):
		"""Method that user pass as functor to DataLoader.

		The method optionally performs neural type checking and add types to the outputs.

		Please note, subclasses of Dataset should not implement `input_types`.

		# Usage:
		dataloader = torch.utils.data.DataLoader(
				....,
				collate_fn=dataset.collate_fn,
				....
		)

		Returns
		-------
			Collated batch, with or without types.
		"""
		# Initialize a dictionary to store the batch data
		batch_dict = {}
		for key in batch[0].keys():
			batch_dict[key] = [torch.tensor(sample[key]) for sample in batch]
			batch_dict[key] = pad_sequence(batch_dict[key], batch_first=True, padding_value=-1)

		return batch_dict
