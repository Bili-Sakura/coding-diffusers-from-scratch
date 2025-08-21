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
# The following code is largely inherited from `huggingface/diffuser` (`src.diffusers.models.unets.unet_2d_condition.py`) but we provide a much more simplified version. For simplicity, we re-use built-in functions and classes from `diffusers` which we will re-implement them later on.
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

from utils import *


class UNet2DConditionModel:

    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,  # latents
        out_channels: int = 4,  # latents
        down_block_types: Tuple[str] = (  # override by config.json
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[
            str
        ] = "UNetMidBlock2DCrossAttn",  # override by config.json
        up_block_types: Tuple[str] = (  # override by config.json
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        mid_block_only_cross_attention: Optional[bool] = None,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        norm_num_groups: Optional[int] = 32,
        time_embedding_type: str = "positional",
        encoder_hid_dim_type: Optional[str] = None,
        encoder_hid_dim: Optional[int] = None,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
        attention_head_dim: Union[
            int, Tuple[int]
        ] = 8,  # the same as 'num_attention_heads' follow `diffusers`
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        attention_type: str = "default",
        act_fn: str = "silu",  # follow `diffusers`
        norm_eps: float = 1e-5,  # follow `diffusers`
    ) -> None:

        super().__init__()

        if num_attention_heads is not None:
            raise ValueError(
                "At the moment it is not possible to define the number of attention heads via `num_attention_heads` because of a naming issue as described in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131. Passing `num_attention_heads` will only be supported in diffusers v0.19."
            )
        # If `num_attention_heads` is not defined (which is the case for most models)
        # it will default to `attention_head_dim`. This looks weird upon first reading it and it is.
        # The reason for this behavior is to correct for incorrectly named variables that were introduced
        # when this library was created. The incorrect naming was only discovered much later in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131
        # Changing `attention_head_dim` to `num_attention_heads` for 40,000+ configurations is too backwards breaking
        # which is why we correct for the naming here.
        # Sakura: LoL :D
        num_attention_heads = num_attention_heads or attention_head_dim

        # Check inputs
        self._check_config(
            down_block_types=self.config.down_block_types,
            up_block_types=self.config.up_block_types,
            only_cross_attention=self.config.only_cross_attention,
            block_out_channels=self.config.block_out_channels,
            layers_per_block=self.config.layers_per_block,
            cross_attention_dim=self.config.cross_attention_dim,
            transformer_layers_per_block=self.config.transformer_layers_per_block,
            reverse_transformer_layers_per_block=self.config.reverse_transformer_layers_per_block,
            attention_head_dim=self.config.attention_head_dim,
            num_attention_heads=self.config.num_attention_heads,
        )

        # input
        conv_in_padding = (self.config.conv_in_kernel - 1) // 2
        # (B, in_channels, H, W) -> (B, block_out_channels[0], H, W)
        self.conv_in = nn.Conv2d(
            in_channels=self.config.in_channels,
            out_channels=self.config.block_out_channels[0],
            kernel_size=self.config.conv_in_kernel,
            # stride=1,
            padding=conv_in_padding,
            # dilation=1,
            # groups=1,
            # bias=True,
            # padding_mode='zeros'
        )
        # time
        # time_embed_dim: int, timestep_input_dim: int
        time_embed_dim, timestep_input_dim = self._set_time_proj(  # TODO
            time_embedding_type=self.config.time_embedding_type,
            block_out_channels=self.config.block_out_channels,
        )
        # (B, timestep_input_dim) -> (B, time_embed_dim)
        self.time_embedding = TimestepEmbedding(
            in_channels=timestep_input_dim,
            time_embed_dim=time_embed_dim,
            # act_fn="silu",
            # out_dim=None,
            # post_act_fn=None,
            # cond_proj_dim=None,
            # sample_proj_bias=True
        )

        self._set_encoder_hid_proj(  # TODO
            encoder_hid_dim_type=self.config.encoder_hid_dim_type,
            cross_attention_dim=cross_attention_dim,
            encoder_hid_dim=self.config.encoder_hid_dim,
        )

        # class embedding, we skip it only focusing on text-to-image generation
        # self._set_class_embedding() # No need to implement

        # self._set_add_embedding() # TODO

        if isinstance(only_cross_attention, bool):
            if mid_block_only_cross_attention is None:
                mid_block_only_cross_attention = only_cross_attention

            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if mid_block_only_cross_attention is None:
            mid_block_only_cross_attention = False

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(
                down_block_types
            )

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        # down
        self.config.out_channels = self.config.block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = self.config.block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type=down_block_type,
                num_layers=layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=self.config.time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=self.config.norm_eps,
                resnet_act_fn=self.config.act_fn,
                transformer_layers_per_block=transformer_layers_per_block[i],
                # resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim[i],
                num_attention_heads=num_attention_heads[
                    i
                ],  # this is an awful naming issue, just provide so that the code not break; we will fix this in our re-implement later on
                # downsample_padding=downsample_padding,
                # dual_cross_attention=dual_cross_attention,
                # use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                # upcast_attention=upcast_attention,
                # resnet_time_scale_shift=resnet_time_scale_shift,
                attention_type=self.config.attention_type,
                # resnet_skip_time_act=resnet_skip_time_act,
                # resnet_out_scale_factor=resnet_out_scale_factor,
                # cross_attention_norm=cross_attention_norm,
                attention_head_dim=(
                    attention_head_dim[i]
                    if attention_head_dim[i] is not None
                    else output_channel
                ),
                # dropout=dropout,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = get_mid_block(
            mid_block_type=self.config.mid_block_type,
            temb_channels=self.config.time_embed_dim,
            in_channels=self.config.block_out_channels[-1],
            resnet_eps=self.config.norm_eps,
            resnet_act_fn=self.config.act_fn,
            # output_scale_factor=1.0,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            num_attention_heads=num_attention_heads[-1],
            cross_attention_dim=cross_attention_dim[-1],
            # dual_cross_attention=False,
            # use_linear_projection=False,
            mid_block_only_cross_attention=mid_block_only_cross_attention,
            # upcast_attention=False,
            # resnet_time_scale_shift="default",
            attention_type=self.config.attention_type,
            # resnet_skip_time_act=False,
            # cross_attention_norm=None,
            attention_head_dim=attention_head_dim[-1],
            # dropout=0.0
        )
        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = (
            list(reversed(transformer_layers_per_block))
            if reverse_transformer_layers_per_block is None
            else reverse_transformer_layers_per_block
        )
        only_cross_attention = list(reversed(only_cross_attention))

        out_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = out_channel
            out_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type=up_block_type,
                num_layers=reversed_layers_per_block[i],
                in_channels=input_channel,
                out_channels=out_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=self.config.time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=self.config.norm_eps,
                resnet_act_fn=self.config.act_fn,
                resolution_idx=i,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                num_attention_heads=reversed_num_attention_heads[i],
                resnet_groups=self.config.norm_num_groups,
                cross_attention_dim=reversed_cross_attention_dim[i],
                # dual_cross_attention=False,
                # use_linear_projection=False,
                only_cross_attention=only_cross_attention[i],
                # upcast_attention=False,
                # resnet_time_scale_shift="default",
                attention_type=self.config.attention_type,
                # resnet_skip_time_act=False,
                # resnet_out_scale_factor=1,
                # cross_attention_norm=None,
                attention_head_dim=(
                    attention_head_dim[i]
                    if attention_head_dim[i] is not None
                    else output_channel
                ),
                # upsample_type=None,
                # dropout=0.0,
            )
            self.up_blocks.append(up_block)

        # out
        if self.config.norm_num_groups is not None:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0],
                num_groups=self.config.norm_num_groups,
                eps=self.config.norm_eps,
            )
            self.conv_out = get_activation(act_fn=self.config.act_fn)

        else:
            self.conv_norm_out = None
            self.conv_act = None

        conv_out_padding = (self.config.conv_out_kernel - 1) // 2
        self.conv_out = nn.Conv2d(
            block_out_channels=block_out_channels[0],
            out_channels=self.config.out_channels,
            kernel_size=self.config.conv_out_kernel,
            padding=self.config.conv_out_padding,
        )

    def _set_add_embedding():
        pass

    def _set_encoder_hid_proj():
        pass

    def _set_time_proj():
        pass

    def get_time_emb():
        pass

    def time_embedding():
        pass

    def process_encoder_hidden_states():
        pass

    def forward(
        self,
        sample: torch.Tensor,  # (B, C, H, W)
        timestep: torch.Tensor,  # (B,)
        encoder_hidden_states: torch.Tensor,  # (B, S, D) text embedding
    ) -> torch.Tensor:
        # 1. time
        # (B, time_embed_dim) or (B, D)
        t_emb = self.get_time_emb(sample=sample, timestep=timestep)  # TODO
        emb = self.time_embedding(t_emb)  # TODO

        # (B, S, D)
        encoder_hidden_states = self.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states
        )  # TODO

        # 2. pre-process
        # (B, block_out_channels[0], H, W)

        sample = self.conv_in(sample)

        # 3. down
        # (B, C, H, W)
        down_block_res_samples = sample

        for downsample_block in self.down_blocks:

            sample, res_samples = downsample_block(
                hidden_states=sample,  # (B, C, H, W)
                temb=emb,  # (B, D)
                encoder_hidden_states=encoder_hidden_states,  # (B, S, D)
            )
            # (B, C, H, W)
            down_block_res_samples += res_samples

        # 4. mid
        # (B, C, H, W)
        sample = self.mid_block(
            hidden_states=sample,  # (B, C, H, W)
            temb=emb,  # (B, D)
        )

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            # (B, C, H, W)
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block)]

            if not is_final_block:
                upsample_size = down_block_res_samples[-1].shape[2:]  # (H,W)

            sample = upsample_block(
                hidden_states=sample,  # (B, C, H, W)
                temb=emb,  # (B, D)
                res_hidden_states_tuple=res_samples,  # (H, W)
            )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)  # (B, C, H, W)
            sample = self.conv_act(sample)  # (B, C, H, W)
        sample = self.conv_out(sample)  # (B, out_channels, H, W)

        # (B, out_channels, H, W)
        # DONE!
        return sample
