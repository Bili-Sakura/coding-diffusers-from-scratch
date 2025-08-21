# Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
# William Peebles and Saining Xie
#
# Copyright (c) 2021 OpenAI
# MIT License
#
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
# The following code is largely inherited from `huggingface/diffuser` (`src.diffusers.models.pipelines.dit.pipeline_dit.py`) but we provide a much more simplified version. For simplicity, we re-use built-in functions and classes from `diffusers` which we will re-implement them later on.

from typing import Dict, List, Optional, Tuple, Union

import torch

from utils import (
    AutoencoderKL,
    KarrasDiffusionSchedulers,
    randn_tensor,
    DiffusionPipeline,
    ImagePipelineOutput,
)

# our implement
from DiT import DiTTransformer2DModel


class DiTPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation based on a Transformer backbone.

    Parameters:
        transformer ([`DiTTransformer2DModel`]): xxx
        vae ([`AutoencoderKL`]): xxx.
        scheduler ([`EulerDiscreteScheduler`]): xxx.
    """

    def __init__(
        self,
        transformer: DiTTransformer2DModel,
        vae: AutoencoderKL,
        scheduler: KarrasDiffusionSchedulers,
        id2label: Optional[
            Dict[int, str]
        ] = None,  # as DiT is default as a class-conditional model.
    ):
        super().__init__()
        self.register_modules(transformer=transformer, vae=vae, scheduler=scheduler)

        # create a imagenet -> id dictionary for easier use
        # we generally store this under `model_index.json`, which contains a key as 'id2label'
        self.labels = {}
        if id2label is not None:
            for key, value in id2label.items():
                for label in value.split(","):
                    self.labels[label.lstrip().rstrip()] = int(key)
            self.labels = dict(sorted(self.labels.items()))

    def get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
        r"""

        Map label strings from ImageNet to corresponding class ids.

        Parameters:
            label (`str` or `dict` of `str`):
                Label strings to be mapped to class ids.

        Returns:
            `list` of `int`:
                Class ids to be processed by pipeline.
        """

        if not isinstance(label, list):
            label = list(label)

        for l in label:
            if l not in self.labels:
                raise ValueError(
                    f"{l} does not exist. Please make sure to select one of the following labels: \n {self.labels}."
                )

        return [self.labels[l] for l in label]

    @torch.no_grad()  # no gradient calculation during inference
    def __call__(
        self,
        class_labels: List[int],
        guidance_scale: float = 4.0,
        generator: Optional[
            Union[torch.Generator, List[torch.Generator]]
        ] = None,  # use List if generated multi-image a time
        num_inference_steps: int = 50,
    ):
        r"""
        The call function to the pipeline for generation.

        Arg:
            class_labels (List[int]): xxx.
            guidance_scale (`float`, *optional*, defaults to 4.0): xxx.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 50): If you use advanced schedule like Euler, DDIM etc., it is enough to use 10-50.
        Examples:

        ```py
        >>> from diffusers import DiTPipeline, EulerDiscreteScheduler
        >>> import torch

        >>> pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
        >>> pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        >>> pipe = pipe.to("cuda")

        >>> # pick words from Imagenet class labels
        >>> pipe.labels  # to print all available words

        >>> # pick words that exist in ImageNet
        >>> words = ["white shark", "umbrella"]

        >>> class_ids = pipe.get_label_ids(words)

        >>> generator = torch.manual_seed(33)
        >>> output = pipe(class_labels=class_ids, num_inference_steps=25, generator=generator)

        >>> image = output.images[0]  # label 'white shark'
        ```

        Returns:
            output: generated image/list of images with format of `PIL`.
        """

        batch_size = len(class_labels)
        latent_size = self.transformer.config.sample_size
        latent_channels = self.transformers.config.in_channels

        # random noise
        latents = randn_tensor(
            shape=(batch_size, latent_channels, latent_size, latent_size),
            generator=generator,
            device=self._execution_device,
            dtype=self.transformer.dtype,
        )
        # classifier-free guidance (cfg)
        # if we use cfg, the latents channel doubles to denote conditioning and unconditioning results respectively
        # details see paper 'Classifier-Free Diffusion Guidance' (NeurIPS 2021)
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents

        class_labels = torch.tensor(
            class_labels, device=self._execution_device
        ).reshape(-1)
        class_null = torch.tensor(
            [1000] * batch_size, device=self._execution_device
        )  # default, we use ImageNet which has 1000 classes
        # classifier-free guidance
        class_labels_input = (
            torch.cat([class_labels, class_null], 0) if guidance_scale > 1 else latents
        )

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.progress_bar(self.scheduler.timesteps):
            if guidance_scale > 1:
                half = latent_model_input[:, len(latent_model_input) // 2]
                latent_model_input = torch.cat([half, half], dim=0)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # we only consider using CPU/GPU, and timesteps is provided
            timesteps = t
            timesteps = timesteps.expand(latent_model_input.shape[0])
            # predict noise model_output [CORE]
            noise_pred = self.transformer(
                latent_model_input,
                timestep=timesteps,
                class_labels=class_labels_input,
            )

            # perform guidance
            if guidance_scale > 1:
                # eps \eps
                eps = noise_pred[:, :latent_channels]
                rest = noise_pred[:, latent_channels:]
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

                # [CORE] for cfg
                half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                eps = torch.cat([half_eps, half_eps], dim=0)

                noise_pred = torch.cat([eps, rest], dim=1)

            # learned sigma
            if self.transformer.config.out_channels // 2 == latent_channels:
                model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
            else:
                # As we previously, only output the same channel as input channel, we do not predict \sigma
                model_output = noise_pred

            # compute previous image: x_t -> x_t-1
            latent_model_input = self.scheduler.step(
                model_output, t, latent_model_input
            ).prev_sample

        if guidance_scale > 1:
            latents, _ = latent_model_input.chunk(2, dim=0)
        else:
            latents = latent_model_input

        # latents -> images
        latents = 1 / self.vae.config.scaling_factor * latents
        samples = self.vae.decode(latents).sample

        # the value range from [-1,1] (used for diffusion) to [0,1] (used for RGB)
        samples = (samples / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()

        samples = self.numpy_to_pil(samples)

        # DONE!
        return samples
