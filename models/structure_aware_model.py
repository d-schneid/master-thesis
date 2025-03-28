from nemo.core import ModelPT
from omegaconf import OmegaConf
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as ptl
from nemo.core.classes.common import PretrainedModelInfo

from data_preprocessing.structure_aware_dataset import StructureAwareDataset


class StructureAwareModel(ModelPT):

	def __init__(self, cfg: OmegaConf, trainer: ptl.Trainer = None):
		super().__init__(cfg=cfg, trainer=trainer)

	@classmethod
	def list_available_models(cls) -> PretrainedModelInfo:
		return None

	def forward(self, code_tokens, text_tokens, ll_sims, ast_leaf_code_token_idxs, lr_paths_types, dfg_node_code_token_idxs, dfg_edges):
		return 0

	def _setup_data_loader(self, cfg):
		dataset = StructureAwareDataset(cfg.data_path)
		return DataLoader(dataset, collate_fn=dataset.collate_fn)

	def setup_training_data(self, train_data_config: OmegaConf):
		self._train_dl = self._setup_data_loader(train_data_config)

	def setup_validation_data(self, val_data_config: OmegaConf):
		self._validation_dl = self._setup_data_loader(val_data_config)

	def setup_test_data(self, test_data_config: OmegaConf):
		self._test_dl = self._setup_data_loader(test_data_config)

	def step_(self, split, batch, batch_idx=None):
		code_tokens, text_tokens, ll_sims, ast_leaf_code_token_idxs, lr_paths_types, dfg_node_code_token_idxs, dfg_edges = batch
		logits = self(code_tokens, text_tokens, ll_sims, ast_leaf_code_token_idxs, lr_paths_types, dfg_node_code_token_idxs, dfg_edges)
		return logits

	def training_step(self, *args, **kwargs):
		return self.step_('train', *args, **kwargs)

	def validation_step(self, *args, **kwargs):
		return self.step_('val', *args, **kwargs)

	def test_step(self, *args, **kwargs):
		return self.step_('test', *args, **kwargs)
