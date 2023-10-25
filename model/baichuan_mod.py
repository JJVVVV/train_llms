import os
from typing import List, Optional, Tuple, Union

import torch
from torch import FloatTensor, LongTensor, Tensor
from transformers.modeling_outputs import CausalLMOutputWithPast

from .modeling_baichuan import BaichuanForCausalLM
from .tricks import shift_embeddings


class BaichuanForCausalLM_shift(BaichuanForCausalLM):
    def __init__(self, config, alpha):
        super().__init__(config)
        self.alpha = alpha

    def forward(
        self,
        input_ids: LongTensor = None,
        attention_mask: Tensor | None = None,
        past_key_values: List[FloatTensor] | None = None,
        inputs_embeds: FloatTensor | None = None,
        labels: LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = False,
        return_dict: bool | None = True,
        **kwargs
    ) -> Tuple | CausalLMOutputWithPast:
        inputs_embs = self.get_input_embeddings()(input_ids)
        inputs_embs = shift_embeddings(inputs_embs, self.alpha)
        # inputs_embs = inputs_embs.to(torch.float16)
        # print(inputs_embs.dtype)
        return super().forward(
            None, attention_mask, past_key_values, inputs_embs, labels, use_cache, output_attentions, output_hidden_states, return_dict, **kwargs
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike | None, alpha, *model_args, **kwargs):
        kwargs["alpha"] = alpha
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return model
