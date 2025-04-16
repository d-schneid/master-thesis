from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from omegaconf import OmegaConf
import torch
import lightning.pytorch as ptl
from nemo.collections import llm
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
import argparse

from Custom_Model.structure_aware_config import StructureAwareConfig
from models.structure_aware_model import StructureAwareModel

#from models.structcoder import StructCoderForConditionalGeneration
#from models.structure_aware_model import StructureAwareModel


if __name__ == '__main__':

	#args_dict = {"model_size": "large", "num_node_types": 25, "max_ast_depth": 10}
	#args = argparse.Namespace(**args_dict)
	#model = StructCoderForConditionalGeneration(args)

	#model = llm.Starcoder2Model.import_from('hf://bigcode/starcoder2-15b')
	#model.load_state_dict()
	#state = model.state_dict()

	#conf = OmegaConf.load('../nemo_configs/model_config.yaml')
	#trainer = ptl.Trainer(devices=1, accelerator='gpu', limit_test_batches=1.0)
	#model = MegatronGPTModel(cfg=conf.model, trainer=trainer)
	#model.restore_from()
	#print(model.state_dict())
	#print("hello")

	#tokenizer = get_nmt_tokenizer(library='huggingface', model_name='bigcode/starcoder2-3b', use_fast=True)
	#model = llm.Starcoder2Model(llm.Starcoder2Config3B(), tokenizer=tokenizer)
	#model.configure_model()
	#config = model.config

	#trainer = ptl.Trainer(devices=1, accelerator='gpu', limit_test_batches=1.0)
	#model = MegatronGPTModel.restore_from('./load_nemo_file/starcoder2-3b.nemo', trainer=trainer)
	#print(model)

	#cfg = OmegaConf.load('../nemo_configs/model_config.yaml')
	#model = StructureAwareModel(cfg=llm.Starcoder2Config15B(), tokenizer=tokenizer)

	if torch.cuda.is_available():
		accelerator = 'gpu'
	else:
		accelerator = 'cpu'

	tokenizer = get_nmt_tokenizer(library='huggingface', model_name='bigcode/starcoder2-3b', use_fast=True)
	model = StructureAwareModel(StructureAwareConfig(), tokenizer=tokenizer)

	trainer = ptl.Trainer(devices=1, accelerator=accelerator, limit_test_batches=1.0)
	trainer.test(model)
