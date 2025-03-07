# Copyright 2024 Bingxin Ke, ETH Zurich. All rights reserved.
# Last modified: 2025-03-07
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
# ---------------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/RollingDepth#-citation
# More information about the method can be found at https://rollingdepth.github.io
# ---------------------------------------------------------------------------------

import logging
from os import PathLike
from typing import Dict, List, Union

import einops
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,  # type: ignore
    DDIMScheduler,  # type: ignore
    DiffusionPipeline,  # type: ignore
    UNet2DConditionModel,  # type: ignore
)
from diffusers.utils import BaseOutput  # type: ignore

from .depth_aligner import DepthAligner
from .video_io import load_video_frames


class RollingDepthOutput(BaseOutput):
    input_rgb: torch.Tensor
    depth_pred: torch.Tensor
    # intermediate results
    snippet_ls: Union[None, List[torch.Tensor]]
    depth_coaligned: Union[None, torch.Tensor]


class RollingDepthPipeline(DiffusionPipeline):
    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215
    N_CHANNEL_PER_LATENT = 4

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: DDIMScheduler,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )

        self.empty_text_embed: torch.Tensor = None  # type: ignore

        logging.debug(f"Pipeline initialized: {type(self)}")

    @torch.no_grad()
    def __call__(
        self,
        # input setting
        input_video_path: PathLike,
        start_frame: int = 0,
        frame_count: int = 0,
        processing_res: int = 1024,
        resample_method: str = "BILINEAR",
        # infer setting
        dilations: List[int] = [1, 25],
        cap_dilation: bool = True,
        snippet_lengths: List[int] = [3],
        init_infer_steps: List[int] = [1],
        strides: List[int] = [1],
        coalign_kwargs: Union[Dict, None] = None,
        refine_step: int = 0,
        refine_snippet_len: int = 3,
        refine_start_dilation: int = 6,
        # other settings
        generator: Union[torch.Generator, None] = None,
        verbose: bool = False,
        max_vae_bs: int = 4,
        unload_snippet: bool = False,
        # output settings
        restore_res: bool = False,
    ) -> RollingDepthOutput:
        assert processing_res >= 0

        # ----------------- Check settings -----------------
        if processing_res > 1024:
            logging.warning(
                f"Procssing at high-resolution ({processing_res}) may lead to suboptimal accuracy."
            )

        # ----------------- Load input data -----------------
        # Load, resize, and normalize input frames
        input_frames, original_res = load_video_frames(
            input_path=input_video_path,
            start_frame=start_frame,
            frame_count=frame_count,
            processing_res=processing_res,
            resample_method=resample_method,
            verbose=verbose,
        )
        input_frames = einops.rearrange(input_frames, "n c h w -> 1 n c h w")
        logging.info(
            f"{input_frames.shape[1]} frames loaded from video {input_video_path}"
        )

        # warn if resize back to big resolution
        if restore_res and max(original_res) > 2048:  # type: ignore
            logging.warning(
                f"Resizing back to large resolution ({list(original_res)}) may result in significant memory usage."  # type: ignore
            )

        # ----------------- Predicting depth -----------------
        pipe_output = self.forward(
            input_frames=input_frames,
            dilations=dilations,
            cap_dilation=cap_dilation,
            snippet_lengths=snippet_lengths,
            init_infer_steps=init_infer_steps,
            strides=strides,
            coalign_kwargs=coalign_kwargs,
            refine_step=refine_step,
            refine_snippet_len=refine_snippet_len,
            refine_start_dilation=refine_start_dilation,
            verbose=verbose,
            generator=generator,
            max_vae_bs=max_vae_bs,
            unload_snippet=unload_snippet,
        )

        torch.cuda.empty_cache()  # clear vram cache

        # ----------------- Resize back -----------------
        if restore_res:
            if verbose:
                logging.info(f"Resizing to the original resolution: {original_res}")
            # Restore RGB resolution
            input_rgb = pipe_output.input_rgb
            input_rgb = resize(
                input_rgb,
                list(original_res),
                interpolation=InterpolationMode.__getitem__(resample_method),
            )
            pipe_output.input_rgb = input_rgb

            # Restore depth resolution
            depth_pred = pipe_output.depth_pred
            depth_pred = resize(
                depth_pred,
                list(original_res),
                interpolation=InterpolationMode.__getitem__(resample_method),
            )
            pipe_output.depth_pred = depth_pred

        return pipe_output

    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    @torch.no_grad()
    def forward(
        self,
        input_frames: torch.Tensor,  # [1 N 3 H W]
        # infer setting
        dilations: List[int],
        cap_dilation: bool,
        snippet_lengths: List[int],
        init_infer_steps: List[int],
        strides: List[int],
        coalign_kwargs: Union[Dict, None],
        refine_step: int,
        refine_snippet_len: int,
        refine_start_dilation: int,
        # other setting
        generator: Union[torch.Generator, None],
        verbose: bool,
        max_vae_bs: int,
        unload_snippet: bool,
    ) -> RollingDepthOutput:
        # ----------------- Check settings -----------------
        assert 1 in dilations, "dilations should include 1"
        # snippet_length
        assert len(snippet_lengths) == len(
            set(snippet_lengths)
        ), f"Repeated values found in {snippet_lengths = }"
        if len(snippet_lengths) > 1:
            assert (
                len(snippet_lengths) == len(dilations)
            ), f"inconsistent lengths of window_size ({snippet_lengths}) and dilations ({dilations})"
        else:
            snippet_lengths = snippet_lengths * len(dilations)
        # infer denoise steps
        if len(init_infer_steps) > 1:
            assert (
                len(init_infer_steps) == len(dilations)
            ), f"inconsistent lengths of init_infer_step ({init_infer_steps}) and dilations ({dilations})"
        else:
            init_infer_steps = init_infer_steps * len(dilations)
        assert min(init_infer_steps) > 0, "Minimum inference step is 1"
        # stride
        if len(strides) > 1:
            assert (
                len(strides) == len(dilations)
            ), f"inconsistent lengths of strides ({strides}) and dilations ({dilations})"
        else:
            strides = strides * len(dilations)
        if [1] * len(dilations) != strides:
            raise NotImplementedError("Only implemented for stride 1")

        # Cap dilation
        seq_len = input_frames.shape[1]
        if cap_dilation:
            for i, dilation in enumerate(dilations):
                dilations[i] = self.cap_max_dilation(
                    seq_len=seq_len,
                    snippet_len=snippet_lengths[i],
                    dilation=dilation,
                    verbose=verbose,
                )
            refine_start_dilation = self.cap_max_dilation(
                seq_len=seq_len,
                snippet_len=refine_snippet_len,
                dilation=refine_start_dilation,
                verbose=verbose,
            )

        # ----------------- Initial prediction -----------------
        device = self.device

        input_frames = input_frames.to(self.dtype).to(device)

        # Encode RGB frames
        rgb_latent = self.encode_rgb(
            input_frames, max_batch_size=max_vae_bs, verbose=verbose
        )

        # Empty text embedding
        if self.empty_text_embed is None:
            self.encode_empty_text()

        B, seq_len, _, h, w = rgb_latent.shape

        if 1 != B:
            raise NotImplementedError("Layered inference is only implemented for B=1")

        torch.cuda.empty_cache()

        # Use the same for every frame
        init_noise = torch.randn(
            (B, 4, h, w),
            device=device,
            dtype=self.dtype,
            generator=generator,
        )
        init_noise = einops.repeat(init_noise, "1 c h w -> B n c h w", n=seq_len, B=B)

        # Get snippets
        snippet_pred_ls = self.init_snippet_infer(
            rgb_latent=rgb_latent,
            init_noise=init_noise,
            dilations=dilations,
            cap_dilation=cap_dilation,
            snippet_lengths=snippet_lengths,
            init_infer_steps=init_infer_steps,
            strides=strides,
            max_vae_bs=max_vae_bs,
            unload_snippet=unload_snippet,
            verbose=verbose,
        )

        # ----------------- Co-alignment -----------------
        coalign_kwargs = {} if coalign_kwargs is None else coalign_kwargs
        depth_aligner = DepthAligner(
            verbose=verbose,
            device=device,
            **coalign_kwargs,
        )
        (depth_coaligned, scales, translations, loss_history) = depth_aligner.run(
            snippet_ls=snippet_pred_ls, dilations=dilations
        )

        # Re-normalize
        depth_coaligned -= depth_coaligned.min()
        depth_coaligned /= depth_coaligned.max()
        depth_coaligned = depth_coaligned * 2.0 - 1.0

        torch.cuda.empty_cache()

        # ----------------- Refinement -----------------
        if refine_step > 0:
            # Encode depth
            depth_latent_coaligned = self.encode_rgb(
                einops.repeat(depth_coaligned, "N 1 H W -> 1 N 3 H W"),
                max_batch_size=max_vae_bs,
                verbose=verbose,
            )
            # Refine
            depth_latent_new = self.refine(
                rgb_latent=rgb_latent,
                depth_latents=depth_latent_coaligned,
                init_noise=init_noise,
                refine_step=refine_step,
                snippet_len=refine_snippet_len,
                start_dilation=refine_start_dilation,
                verbose=verbose,
            )
            # Decode
            depth_pred = self.decode_depth(
                depth_latent_new, max_batch_size=max_vae_bs, verbose=verbose
            )
        else:
            depth_pred = depth_coaligned

        # ----------------- Output -----------------
        pipe_out = RollingDepthOutput(
            input_rgb=input_frames.detach().cpu().squeeze(0) / 2.0 + 0.5,
            depth_pred=depth_pred.detach().cpu().squeeze(0),
            snippet_ls=[snippet.detach().cpu() for snippet in snippet_pred_ls],
            depth_coaligned=depth_coaligned.detach().cpu().squeeze(0),
        )
        return pipe_out

    def init_snippet_infer(
        self,
        rgb_latent: torch.Tensor,
        init_noise: torch.Tensor,
        dilations: List[int],
        cap_dilation: bool,
        snippet_lengths: List[int],
        init_infer_steps: List[int],
        strides: List[int],
        max_vae_bs: int,
        unload_snippet: bool,
        verbose: bool,
    ) -> List[torch.Tensor]:
        device = self.device

        B, seq_len, _, h, w = rgb_latent.shape  # latent shape
        if 1 != B:
            raise NotImplementedError("RollingDepth is implemented for B=1")

        # Empty text embedding
        batch_empty_text_embed = self.empty_text_embed.to(device)  # [1, 2, 1024]

        # Output
        snippet_pred_ls = []

        # >>> Go through dilations >>>
        iterable_init_infer = zip(dilations, snippet_lengths, strides, init_infer_steps)
        if verbose:
            iterable_init_infer = tqdm(
                iterable_init_infer,
                desc=" Initial snippet inference",
                leave=False,
                total=len(dilations),
            )
        for dilation, snippet_len, stride, init_infer_step in iterable_init_infer:
            # Set timesteps
            self.scheduler.set_timesteps(init_infer_step, device=device)
            timesteps = self.scheduler.timesteps  # [T]

            # Indice of snippet frames
            snippet_idx_ls = self.get_snippet_indice(
                i_step=0,
                timesteps=timesteps,
                seq_len=seq_len,
                snippet_len=snippet_len,
                dilation_start=dilation,
                dilation_end=dilation,
                stride=stride,
            )

            # >> Go through snippets >>
            depth_snippet_latent_ls = []
            snippet_iterable = snippet_idx_ls
            if verbose:
                snippet_iterable = tqdm(
                    snippet_iterable,
                    desc=f"{' '*2}Predicting snippets with dilation {dilation}",
                    leave=False,
                )
            for snippet_idx in snippet_iterable:
                # Get input frames
                rgb_latent_snippet = rgb_latent[:, snippet_idx, :, :, :].clone()
                depth_latent_snippet = init_noise[:, snippet_idx, :, :, :].clone()

                # Reset timesteps
                self.scheduler.set_timesteps(init_infer_step, device=device)
                timesteps = self.scheduler.timesteps  # [T]

                # Denoising loop
                iterable_step = timesteps
                if verbose and init_infer_step > 1:
                    iterable_step = tqdm(
                        iterable_step,
                        total=len(timesteps),
                        leave=False,
                        desc=f"{' '*3}Denoising",
                    )
                for t_current in iterable_step:
                    t_input = t_current.repeat(rgb_latent_snippet.shape[1])
                    # Denoising step
                    noise_pred = self.single_step(
                        rgb_latent=rgb_latent_snippet,
                        depth_latent=depth_latent_snippet,
                        timestep=t_input,
                        encoder_hidden_states=batch_empty_text_embed,
                    )
                    _scheduler_output = self.scheduler.step(
                        noise_pred, t_current, depth_latent_snippet
                    )
                    depth_latent_snippet = _scheduler_output.prev_sample
                depth_snippet_latent_ls.append(depth_latent_snippet)
            # << Go through snippets <<
            depth_snippet_latent = torch.concat(depth_snippet_latent_ls, dim=0)

            # Decode to depth
            del depth_snippet_latent_ls
            triplets_decoded = self.decode_depth(
                depth_snippet_latent,
                max_batch_size=max_vae_bs,
                verbose=verbose,
                unload_snippet=unload_snippet,
            )

            snippet_pred_ls.append(triplets_decoded)
            torch.cuda.empty_cache()
        # <<< Go through dilations <<<
        return snippet_pred_ls

    @staticmethod
    def get_snippet_indice(
        i_step: int,
        timesteps: torch.Tensor,
        seq_len: int,
        snippet_len: int,
        dilation_start: int,
        dilation_end: int,
        stride: int,
    ) -> List[List[int]]:
        gap_start = dilation_start - 1
        gap_end = dilation_end - 1
        assert (
            gap_start >= gap_end
        ), f"expect gap_start > gap_end, but got {gap_start} and {gap_end}"
        assert gap_start >= 0 and gap_end >= 0

        total_step = len(timesteps)
        gap_cur = int((1 - (i_step) / total_step) * (gap_start - gap_end) + gap_end)

        # Generate snippet indice
        snippet_idx_ls = []
        total_window_size = (snippet_len - 1) * (gap_cur + 1) + 1
        # index of the first frame
        i_start_ls = list(range(0, seq_len - total_window_size + 1, stride))
        # last window (for stride > 1)
        if i_start_ls[-1] < seq_len - total_window_size:
            i_start_ls.append(seq_len - total_window_size)
        for i_start in i_start_ls:
            input_idx = list(range(i_start, i_start + total_window_size, gap_cur + 1))
            snippet_idx_ls.append(input_idx)

        # Check if every frame is covered
        if not set(range(0, seq_len)) == set([x for f in snippet_idx_ls for x in f]):
            logging.warning(
                "Not every frame is covered. Consider reducing dilation for short videos"
            )
        return snippet_idx_ls

    @staticmethod
    def cap_max_dilation(seq_len: int, snippet_len: int, dilation: int, verbose: bool):
        # Cap by sequence_len
        max_allowed_gap = int(seq_len / snippet_len) - 1
        if max_allowed_gap < dilation:
            temp_msg = f"{dilation = } is too big for {seq_len} frames. Reduced to {max_allowed_gap}"
            if verbose:
                logging.info(temp_msg)
            else:
                logging.debug(temp_msg)
            dilation = min(max_allowed_gap, dilation)
        return dilation

    def refine(
        self,
        rgb_latent: torch.Tensor,
        depth_latents: torch.Tensor,
        init_noise: torch.Tensor,
        refine_step: int,
        snippet_len: int,
        start_dilation: int,
        verbose: bool,
        skip_t_ratio: float = 0.5,
    ) -> torch.Tensor:
        device = self.device
        B, seq_len, _, h, w = rgb_latent.shape  # latent shape

        if 1 != B:
            raise NotImplementedError("Layered inference is only implemented for B=1")

        # Set timesteps
        total_scheduler_step = int(refine_step / skip_t_ratio)
        assert total_scheduler_step <= self.scheduler.config.get(
            "num_train_timesteps"
        ), "Too many refinement steps"
        self.scheduler.set_timesteps(total_scheduler_step, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # Start from intermediate step
        _start_idx = int(len(timesteps) * skip_t_ratio)
        timesteps = timesteps[_start_idx:]
        assert 0 < len(timesteps) < total_scheduler_step, f"invalid {skip_t_ratio = }"

        # Add noise to latent
        depth_latent_new = self.scheduler.add_noise(
            original_samples=depth_latents,
            noise=init_noise.clone().to(device, dtype=self.dtype),
            timesteps=timesteps[0],
        )

        # Empty text embedding
        batch_empty_text_embed = self.empty_text_embed.to(device)  # [1, 2, 1024]

        # Timestep of each frame
        frame_timestep = (
            torch.zeros((seq_len), device=device, dtype=timesteps.dtype) + timesteps[0]
        )

        # Denoising loop
        iterable_step = enumerate(timesteps)
        if verbose:
            iterable_step = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 2 + "Diffusion denoising",
            )
        for i_step, _ in iterable_step:
            snippet_idx_ls = self.get_snippet_indice(
                i_step=i_step,
                timesteps=timesteps,
                seq_len=seq_len,
                snippet_len=snippet_len,
                dilation_start=start_dilation,
                dilation_end=1,
                stride=1,
            )

            t_current = timesteps[i_step]
            t_next = timesteps[i_step + 1] if i_step < len(timesteps) - 1 else None

            # Clear up new latent
            depth_latent_old = depth_latent_new.clone().detach()
            depth_latent_new *= 0.0
            count_depth_latent_new = torch.zeros(
                (*depth_latent_new.shape[:2], 1, 1, 1),
                device=depth_latent_new.device,
            ).int()  # [B, N, 1, 1, 1]

            # >>> Iterate through snippets >>>
            iterabel_snippet = snippet_idx_ls
            if verbose:
                iterabel_snippet = tqdm(
                    iterabel_snippet,
                    desc=" " * 4 + f"step {i_step+1} sliding windows",
                    leave=False,
                )
            for snippet_idx in iterabel_snippet:
                # Get input frames
                rgb_latent_input = rgb_latent[:, snippet_idx, :, :, :]
                # Use old latent
                depth_latent_input = depth_latent_old[:, snippet_idx, :, :, :]
                t_input = t_current.repeat(len(snippet_idx))

                # Denoising step
                noise_pred = self.single_step(
                    rgb_latent=rgb_latent_input,
                    depth_latent=depth_latent_input,
                    timestep=t_input,
                    encoder_hidden_states=batch_empty_text_embed,
                )
                _scheduler_output = self.scheduler.step(
                    noise_pred, t_current, depth_latent_input
                )
                depth_latent_pred = _scheduler_output.prev_sample

                # Cumulate new latents
                depth_latent_new[
                    :, snippet_idx, :, :, :
                ] += depth_latent_pred.clone().detach()
                count_depth_latent_new[:, snippet_idx, 0, 0, 0] += 1
            # <<< Iterate through snippets <<<

            # Average new latents
            assert torch.all(count_depth_latent_new > 0)
            depth_latent_new = depth_latent_new / count_depth_latent_new
            if t_next is not None:
                frame_timestep[:] = t_next

        return depth_latent_new

    def single_step(
        self,
        rgb_latent: torch.Tensor,
        depth_latent: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        assert depth_latent.shape == rgb_latent.shape
        num_view = rgb_latent.shape[1]
        rgb_latent = einops.rearrange(rgb_latent, "b n c h w -> (b n) c h w")
        depth_latent = einops.rearrange(depth_latent, "b n c h w -> (b n) c h w")

        # Concat rgb and depth latents
        unet_input = torch.cat([rgb_latent, depth_latent], dim=1)  # [N, 8, h, w]

        # predict the noise residual
        noise_pred = self.unet(
            unet_input,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            num_view=num_view,
        ).sample  # [(B N) 4 h w]

        noise_pred = einops.rearrange(
            noise_pred, "(B N) C h w -> B N C h w", N=num_view
        )

        return noise_pred

    def encode_rgb(
        self, rgb_in: torch.Tensor, max_batch_size: int, verbose: bool = False
    ) -> torch.Tensor:
        self.vae = self.vae.to(self.device)
        if 5 == rgb_in.dim():
            B, N, _, H, W = rgb_in.shape
        else:
            B, _, H, W = rgb_in.shape
            N = 1

        rgb_in = einops.rearrange(rgb_in, "B N C H W -> (B N) C H W")

        # Process in batches
        latents = []
        iterable = range(0, B * N, max_batch_size)
        if verbose:
            iterable = tqdm(
                iterable,
                total=len(list(iterable)),
                leave=False,
                desc=" " * 4 + "Encoding",
            )
        for i in iterable:
            batch = rgb_in[i : i + max_batch_size]
            # encode
            h = self.vae.encoder(batch)
            moments = self.vae.quant_conv(h)
            mean, logvar = torch.chunk(moments, 2, dim=1)
            latents.append(mean)

        # Concatenate all batches
        rgb_latent = torch.cat(latents, dim=0)

        # scale latent
        rgb_latent = rgb_latent * self.rgb_latent_scale_factor

        h, w = rgb_latent.shape[-2:]  # latent shape
        rgb_latent = einops.rearrange(rgb_latent, "(B N) c h w -> B N c h w", B=B)

        return rgb_latent

    def decode_depth(
        self,
        depth_latent: torch.Tensor,
        max_batch_size: int,
        verbose: bool = False,
        unload_snippet: bool = False,
    ) -> torch.Tensor:
        self.vae = self.vae.to(self.device)

        B, N, C, H, W = depth_latent.shape

        depth_latent = einops.rearrange(depth_latent, "b n c h w -> (b n) c h w")

        # scale latent
        depth_latent = depth_latent / self.depth_latent_scale_factor

        # Process in batches
        decoded_outputs = []
        iterable = range(0, B * N, max_batch_size)
        if verbose:
            iterable = tqdm(
                iterable,
                total=len(list(iterable)),
                leave=False,
                desc=" " * 4 + "Decoding",
            )
        for i in iterable:
            batch = depth_latent[i : i + max_batch_size]
            # decode
            z = self.vae.post_quant_conv(batch)
            stacked = self.vae.decoder(z)
            if unload_snippet:
                stacked = stacked.cpu()
            decoded_outputs.append(stacked)
            torch.cuda.empty_cache()  # clear vram cache
        all_decoded = torch.cat(decoded_outputs, dim=0)

        # mean of output channels
        depth_mean = all_decoded.mean(dim=1, keepdim=True)
        depth_mean = einops.rearrange(depth_mean, "(b n) c h w -> b n c h w", n=N)

        return depth_mean
