from pathlib import Path

from structure_aware_customization.model.structure_aware_starcoder2_model import StructureAwareStarcoder2Model
from structure_aware_customization.model.structure_aware_starcoder2_config import StructureAwareStarcoder2Config

from nemo.collections import llm
from nemo import lightning as nl


if __name__ == "__main__":
	model = StructureAwareStarcoder2Model(StructureAwareStarcoder2Config())
	llm.import_ckpt(model=model, source='hf://local_hf_dir', output_path=Path('output_dir'), overwrite=True)
	trainer = nl.Trainer()
	trainer.test(model=model)
