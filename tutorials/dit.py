# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

######################################################################
# Copyright 2025 Sakura. All rights reserved.
# The following code is largely inherited from `huggingface/diffuser` (`src.diffusers.models.transformers.dit_transformer_2d.py`) but we provide a much more simplified version.

# Sakura: For simplicity, we re-use built-in functions and classes from `diffusers` which we will re-implement later on.

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from utils import (
    logging,
    BasicTransformerBlock,
    PatchEmbed,
    Transformer2DModelOutput,
    ModelMixin,
    ConfigMixin,
    register_to_config,
)


# DiT Transformers 2D
class DiTTransformer2DModel(ModelMixin, ConfigMixin):
    r"""
    A 2D Transformer model as introduced in DiT (https://huggingface.co/papers/2212.09748). Our default configuration is inherited from DiT-XL/2.

    Examples:
    ```py
    >>> transformer = DiTTransformer2DModel()
    >>> noise_pred = self.transformer(
                latent_model_input,
                timestep,
                class_labels,
            )[0]
    ```

    """

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        sample_size: int = 32,
        in_channels: int = 4,  # latents
        out_channels: int = 4,  # we only consider modeling noise, omit \sigma
        num_attention_heads: int = 16,
        attention_head_dim: int = 72,
        num_embeds_ada_norm: int = 1000,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        norm_type: str = "ada_norm_zero",  # default in DiT implement
        norm_elementwise_affine: bool = False,
        upcast_attention: bool = False,
        num_layers: int = 28,
    ):
        super().__init__()

        self.inner_dim = (
            self.config.num_attention_heads * self.config.attention_head_dim
        )
        self.pos_embed = PatchEmbed(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim,
        )

        self.transformers_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    # dropout=0.0,
                    # cross_attention_dim=None,
                    activation_fn=self.config.activation_fn,
                    num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                    attention_bias=self.config.attention_bias,
                    # only_cross_attention=False,
                    # double_self_attention=False,
                    upcast_attention=self.config.upcast_attention,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_type=self.config.norm_type,
                    # norm_eps= 1e-5,
                    # final_dropout=False,
                    # attention_type="default",
                    # positional_embeddings=None,
                    # num_positional_embeddings=None,
                    # ada_norm_continous_conditioning_embedding_dim=None,
                    # ada_norm_bias=None,
                    # ff_inner_dim=None,
                    # ff_bias=True,
                    # attention_out_bias=True
                )
                for _ in range(self.config.num_layers)
            ]
        )

        self.proj_out1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.norm_out = nn.LayerNorm(
            self.inner_dim,
            elementwise_affine=False,
            eps=1e-6,  # follow diffusers implement
        )
        self.proj_out2 = nn.Linear(
            self.inner_dim, self.config.patch_size * self.config.out_channels
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        class_labels: Optional[
            torch.LongTensor
        ] = None,  # optional, None for unconditional generation
    ):
        """
        Processes hidden states through transformer blocks

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, channel, heigh, width)` continuous value): ...
            timestep (`torch.LongTensor`): indicate denoising step.
            class_labels (`torch.LongTensor` of shape `(batch_size, class_id)`): ...

        Return:
            output (`torch.LongTensor` of shape `(batch, out_channels, H, W)`): ...

        """

        # 1. Input
        # hidden_states: (batch, in_channels, H, W)
        height = hidden_states.shape[-2] // self.patch_size
        width = hidden_states.shape[-1] // self.patch_size
        # (batch, in_channels, H, W) -> (batch, num_patches, inner_dim)
        hidden_states = self.pos_embed(hidden_states)

        # 2. Blocks
        for block in self.transformers_blocks:
            hidden_states = block(
                hidden_states,  # (batch, num_patches, inner_dim)
                timestep,
                class_labels,
            )

        # hidden_states: (batch, num_patches, inner_dim)

        # 3. Output
        # conditioning: (batch, inner_dim)
        conditioning = self.transformers_blocks[0].norm1.emb(
            timestep,
            class_labels,
        )
        # shift: (batch, inner_dim), scale: (batch, inner_dim)
        shift, scale = self.proj_out1(F.silu(conditioning)).chunk(2, dim=1)
        # hidden_states: (batch, num_patches, inner_dim)
        hidden_states = (
            self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        )
        # hidden_states: (batch, num_patches, patch_size*patch_size*out_channels)
        hidden_states = self.proj_out2(hidden_states)

        # 3'. Unpatchify
        # num_patches =  hidden_states.shape[1] = width**2 = width**2
        height = width = int(
            hidden_states.shape[1] ** 0.5
        )  # notice, we only focus on square input; if you with non-square, things go complex
        # hidden_states: (batch, num_patches, patch_size*patch_size*out_channels) -> (batch, height, width, patch_size, patch_size, out_channels)
        hidden_states = hidden_states.reshape(
            shape=(
                -1,
                height,
                width,
                self.config.patch_size,
                self.config.patch_size,
                self.config.out_channels,
            )
        )

        # Rearrange axes (batch, height, width, patch_size, patch_size, out_channels) to (batch, out_channels, height, patch_size, width, patch_size)
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)

        # (batch, out_channels, H, W)
        output = hidden_states.reshape(
            shape=(
                -1,
                self.config.out_channels,
                height * self.config.patch_size,
                width * self.config.patch_size,
            )
        )
        # DONE!
        return output
