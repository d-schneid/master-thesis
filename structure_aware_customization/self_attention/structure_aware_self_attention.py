from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig


class StructureAwareSelfAttention(SelfAttention):

	def __init__(
			self,
			config: TransformerConfig,
			submodules: SelfAttentionSubmodules,
			layer_number: int,
			attn_mask_type=AttnMaskType.padding,
			cp_comm_type: str = None,
	):
		super().__init__(
			config=config,
			submodules=submodules,
			layer_number=layer_number,
			attn_mask_type=attn_mask_type,
			cp_comm_type=cp_comm_type
		)
		# TODO: init embeddings/weights that are used in self_attention
		# alternatively define embeddings/weights in own MCoreGPTModel and input them as params to structure_aware_layer_spec
		# but better define them co-located with the self_attention

	def forward(
			self,
			hidden_states,
			attention_mask,
			key_value_states=None,
			inference_params=None,
			rotary_pos_emb=None,
			rotary_pos_cos=None,
			rotary_pos_sin=None,
			attention_bias=None,
			packed_seq_params=None,
			sequence_len_offset=None,
	):

		return super().forward(
			hidden_states=hidden_states,
			attention_mask=attention_mask,
			key_value_states=key_value_states,
			inference_params=inference_params,
			rotary_pos_emb=rotary_pos_emb,
			rotary_pos_cos=rotary_pos_cos,
			rotary_pos_sin=rotary_pos_sin,
			attention_bias=attention_bias,
			packed_seq_params=packed_seq_params,
			sequence_len_offset=sequence_len_offset,
		)
