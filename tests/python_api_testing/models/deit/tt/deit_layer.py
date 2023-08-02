from typing import Optional, Set, Tuple, Union

import torch
from torch import nn

import tt_lib
from tt_lib.fallback_ops import fallback_ops
from tests.python_api_testing.models.deit.tt.deit_config import DeiTConfig
from tests.python_api_testing.models.deit.tt.deit_attention import TtDeiTAttention
from tests.python_api_testing.models.deit.tt.deit_intermediate import TtDeiTIntermediate
from tests.python_api_testing.models.deit.tt.deit_output import TtDeiTOutput

class TtDeiTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: DeiTConfig(), device, state_dict=None, base_address="") -> None:
        super().__init__()

        self.attention = TtDeiTAttention(config, device, state_dict, base_address = f'{base_address}.attention')
        self.intermediate = TtDeiTIntermediate(config, device, state_dict, base_address = f'{base_address}.intermediate')
        self.output = TtDeiTOutput(config, device, state_dict, base_address = f'{base_address}.output')

        ln_bw = state_dict[f"{base_address}.layernorm_before.weight"]
        ln_bb = state_dict[f"{base_address}.layernorm_before.bias"]
        self.layernorm_before = fallback_ops.LayerNorm(normalized_shape=config.hidden_size,
                                                        weights=ln_bw,
                                                        biases=ln_bb,
                                                        eps=config.layer_norm_eps)

        ln_aw = state_dict[f"{base_address}.layernorm_after.weight"]
        ln_ab = state_dict[f"{base_address}.layernorm_after.bias"]
        self.layernorm_after = fallback_ops.LayerNorm(normalized_shape=config.hidden_size,
                                                        weights=ln_aw,
                                                        biases=ln_ab,
                                                        eps=config.layer_norm_eps)


    def forward(
        self,
        hidden_states: tt_lib.tensor.Tensor,
        head_mask: Optional[tt_lib.tensor.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[tt_lib.tensor.Tensor, tt_lib.tensor.Tensor], Tuple[tt_lib.tensor.Tensor]]:

        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in DeiT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = tt_lib.tensor.add(attention_output , hidden_states)

        # in DeiT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs
