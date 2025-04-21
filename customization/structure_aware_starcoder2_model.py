from nemo.collections import llm
from nemo.collections.llm import Starcoder2Model, Qwen2Model
from torch import nn
from typing import Annotated, Callable, Optional
from nemo.lightning import OptimizerModule
from nemo.collections.llm.utils import Config
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
	from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


class StructureAwareStarcoder2Model(Qwen2Model):

	def __init__(self,
				 config: Annotated[Optional[llm.Qwen2Config], Config[llm.Qwen2Config]] = None,
				 optim: Optional[OptimizerModule] = None,
				 tokenizer: Optional["TokenizerSpec"] = None,
				 model_transform: Optional[Callable[[nn.Module], nn.Module]] = None
	):
		super().__init__(config=config, optim=optim, tokenizer=tokenizer, model_transform=model_transform)

	def forward(
        self,
        input_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        decoder_input: Optional[torch.Tensor] = None,
        inference_params=None,
        packed_seq_params=None,
		code_tokens = None,
		code_tokens_pos_ids=None,
		text_tokens = None,
		text_tokens_pos_ids=None,
		ll_sims = None,
		ast_leaf_code_token_idxs = None,
		lr_paths_types = None,
		lr_paths_len = None,
		dfg_node_code_token_idxs = None,
		dfg_edges = None,
		dfg_node_mask = None,
    ):
		output_tensor = self.module(input_ids, position_ids, attention_mask, labels=labels, code_tokens=code_tokens,
					code_tokens_pos_ids=code_tokens_pos_ids, text_tokens=text_tokens,
					text_tokens_pos_ids=text_tokens_pos_ids, ll_sims=ll_sims,
					ast_leaf_code_token_idxs=ast_leaf_code_token_idxs, lr_paths_types=lr_paths_types,
					lr_paths_len=lr_paths_len, dfg_node_code_token_idxs=dfg_node_code_token_idxs,dfg_edges=dfg_edges,
					dfg_node_mask=dfg_node_mask)

		return output_tensor
