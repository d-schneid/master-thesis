from omegaconf import OmegaConf
import torch
import lightning.pytorch as ptl

from models.structure_aware_model import StructureAwareModel


if __name__ == '__main__':
	cfg = OmegaConf.load('../nemo_configs/model_config.yaml')
	model = StructureAwareModel(cfg=cfg.model)

	if torch.cuda.is_available():
		accelerator = 'gpu'
	else:
		accelerator = 'cpu'

	trainer = ptl.Trainer(devices=1, accelerator=accelerator, limit_test_batches=1.0)
	trainer.test(model)
