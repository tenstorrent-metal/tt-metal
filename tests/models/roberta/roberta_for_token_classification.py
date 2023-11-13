# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
import sys
from typing import Optional, Tuple, Union, List
import torch
import torch.nn as nn
import numpy as np
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from tests.models.roberta.roberta_common import (
    torch2tt_tensor,
    tt2torch_tensor,
)
from tests.models.roberta.roberta_model import TtRobertaModel

import tt_lib
from tt_lib.fallback_ops import fallback_ops

from dataclasses import dataclass


@dataclass
class TtTokenClassifierOutput:
    loss: tt_lib.tensor.Tensor = None
    logits: tt_lib.tensor.Tensor = None
    hidden_states: tt_lib.tensor.Tensor = None
    attentions: tt_lib.tensor.Tensor = None


class TtRobertaForTokenClassification(nn.Module):
    def __init__(self, config, state_dict, base_address, device, reference_model):
        super().__init__()
        self.mem_config = tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferStorage.L1)
        self.config = config
        self.device = device
        self.num_labels = config.num_labels

        self.roberta = TtRobertaModel(
            config=config,
            state_dict=state_dict,
            base_address=f"{base_address}" + "roberta",
            device=device,
            reference_model=reference_model.roberta,
            add_pooling_layer=False,
        )

        # TODO: Add when implementing training
        # classifier_dropout = (
        #     config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        # )
        # self.dropout = nn.Dropout(classifier_dropout)

        self.classifier_weight = torch2tt_tensor(
            state_dict[f"classifier.weight"], self.device
        )
        self.classifier_bias = torch2tt_tensor(
            state_dict[f"classifier.bias"], self.device
        )

    def linear(self, x, weight, bias):
        weight = tt_lib.tensor.transpose(weight)
        x = tt_lib.tensor.matmul(x, weight, self.mem_config)
        if bias is not None:
            x = tt_lib.tensor.bcast(
                x,
                bias,
                tt_lib.tensor.BcastOpMath.ADD,
                tt_lib.tensor.BcastOpDim.H,
                self.mem_config,
            )
        return x

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[tt_lib.tensor.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[tt_lib.tensor.Tensor] = None,
        inputs_embeds: Optional[tt_lib.tensor.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[tt_lib.tensor.Tensor], TtTokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs.last_hidden_state

        # TODO: Add when implementing training
        # sequence_output = self.dropout(sequence_output)
        logits = self.linear(
            sequence_output, self.classifier_weight, self.classifier_bias
        )

        loss = None
        if labels is not None:
            # TODO: Training not supported for now.
            pass

            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TtTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
