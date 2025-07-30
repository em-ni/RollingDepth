# Copyright 2024 Dominik Narnhofer, Bingxin Ke, ETH Zurich. All rights reserved.
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


import torch
from torch.optim.adam import Adam
import logging
from typing import List, Tuple
from tqdm import tqdm


class DepthAligner:
    def __init__(
        self,
        device: torch.device,
        factor: int = 10,
        lmda: float = 1e-1,
        lmda2: float = 1e-1,
        lmda3: float = 1e1,
        lr: float = 1e-3,
        num_iterations: int = 2000,
        border: int = 2,
        verbose: bool = False,
        depth_loss_weight: float = 1.0,
        loss_scale=1.0,
    ):
        self.factor = factor  # Depth down-scale factor for s,t computation
        self.lmda = lmda  # Controls soft constraints on s and t
        self.lr = lr  # Optimizer step size
        self.num_iterations = num_iterations  # Optimizer iterations
        self.border = border  # Num pixels for border crop
        self.verbose = verbose
        self.device = device
        self.lmda2 = lmda2
        self.depth_loss_weight = depth_loss_weight
        self.loss_scale = loss_scale
        self.lmda3 = lmda3

    # Create indices to keep data structures simple
    def create_triplet_indices(self, sequence_length: int, gap: int, window_size: int):
        gap += 1  # Adjust gap for inclusive indexing
        index_list = []

        for i in range(sequence_length - (window_size - 1) * gap):
            indices = [i + j * gap for j in range(window_size)]
            index_list.append(indices)

        indices = torch.tensor(index_list)
        return indices

    def run(self, snippet_ls: List[torch.Tensor], dilations: List[int]):
        device = self.device
        snippet_lenghts = [snippet.shape[1] for snippet in snippet_ls]
        gaps = [d - 1 for d in dilations]
        sequence_length = (
            len(snippet_ls[0])
            + (snippet_lenghts[0] - 1) * gaps[0]
            + (snippet_lenghts[0] - 1)
        )

        mn = min([snippet.min() for snippet in snippet_ls])  # type: ignore
        snippet_ls = [tmp - mn for tmp in snippet_ls]

        # Exclude border artifact
        triplets_scaled = [
            tmp[:, :, :, self.border : -self.border, self.border : -self.border].to(
                device
            )
            for tmp in snippet_ls
        ]

        # Scale down to factor size
        triplets_scaled = [
            tmp[:, :, :, :: self.factor, :: self.factor] for tmp in triplets_scaled
        ]

        # Create triplet indices
        indices_list = [
            self.create_triplet_indices(sequence_length, g, w)
            for g, w in zip(gaps, snippet_lenghts)
        ]

        scales, translations, loss_history = self.optimize(
            snippet_ls=triplets_scaled,
            indices_list=indices_list,
            sequence_length=sequence_length,
        )

        merged_scaled_triplets = self.merge_scaled_triplets(
            snippet_ls=snippet_ls,
            indices_list=indices_list,
            s_list=scales,
            t_list=translations,
            sequence_length=sequence_length,
            # device=torch.device("cpu"),
            device=torch.device("cuda"),
        )

        return (
            merged_scaled_triplets,
            scales,
            translations,
            loss_history,
        )

    # Scaling Optimizer
    def optimize(
        self,
        snippet_ls: List[torch.Tensor],
        indices_list: List[torch.Tensor],
        sequence_length: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Tuple]]:
        device = self.device
        H, W = snippet_ls[0].shape[-2], snippet_ls[0].shape[-1]
        windows = [triplet.shape[1] for triplet in snippet_ls]

        snippet_ls = [
            triplet.reshape(triplet.shape[0], w, H * W)
            for triplet, w in zip(snippet_ls, windows)
        ]
        scales = [
            torch.ones(A.shape[0], 1, 1, device=device, requires_grad=True)
            for A in snippet_ls
        ]
        translations = [
            torch.zeros(A.shape[0], 1, 1, device=device, requires_grad=True)
            for A in snippet_ls
        ]

        optimizer = Adam(scales + translations, lr=self.lr, betas=(0.5, 0.9))

        loss_ls = []

        def closure():
            A_scaled = [
                reshaped_tensor * s + t
                for reshaped_tensor, s, t in zip(snippet_ls, scales, translations)
            ]

            M = torch.cat(
                [torch.zeros(w, sequence_length, H * W, device=device) for w in windows]
            )
            M_depth = torch.cat(
                [torch.zeros(w, sequence_length, H * W, device=device) for w in windows]
            )
            B = torch.cat(
                [torch.zeros(w, sequence_length, H * W, device=device) for w in windows]
            )

            for i, (scaled_tensor, indices, w) in enumerate(
                zip(A_scaled, indices_list, windows)
            ):
                M_depth[torch.arange(i * w, (i + 1) * w)[:, None], indices.long().T] = (
                    scaled_tensor.clip(1e-3).permute(1, 0, 2)
                ) ** -1
                M[
                    torch.arange(i * w, (i + 1) * w)[:, None], indices.long().T
                ] = scaled_tensor.permute(1, 0, 2)
                B[torch.arange(i * w, (i + 1) * w)[:, None], indices.long().T] = 1

            summ = M.sum(0) / B.sum(0)
            summ_depth = M_depth.sum(0) / B.sum(0)

            # Calculate target
            with torch.no_grad():
                target = summ.clone().detach()
                target_depth = summ_depth.clone().detach()
                scale = target.abs().mean(-1, keepdim=True)
                scale_depth = target_depth.abs().mean(-1, keepdim=True)

            loss = torch.abs((M - target) * B / scale).mean()
            loss_depth = torch.abs((M_depth - target_depth) * B / scale_depth).mean()

            loss = (loss + self.depth_loss_weight * loss_depth).mean()

            soft_constraints = sum(
                self.lmda2 * (torch.max(torch.tensor(0.0), 1 - s) ** 2).mean()
                + self.lmda3 * (t**2).mean()
                for s, t in zip(scales, translations)
            )

            loss = self.loss_scale * loss + soft_constraints
            loss.backward()
            loss_ls.append((loss.item(), summ.min().item(), summ.max().item()))
            return loss

        # Optimization loop
        iterable = range(self.num_iterations)
        if self.verbose:
            iterable = tqdm(iterable, desc="Co-align snippets", leave=False)
        for i in iterable:
            optimizer.zero_grad()
            optimizer.step(closure)

            if i % 10 == 0 and self.verbose:
                logging.debug(
                    f"Iteration {i}, Loss_diff: {loss_ls[-1][0]:.6f}, Min: {loss_ls[-1][1]:.6f}, Max: {loss_ls[-1][2]:.6f}"
                )

        return scales, translations, loss_ls

    def merge_scaled_triplets(
        self,
        snippet_ls: List[torch.Tensor],
        indices_list: List[torch.Tensor],
        s_list: List[torch.Tensor],
        t_list: List[torch.Tensor],
        sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        snippet_ls = [a.to(device) for a in snippet_ls]
        dtype = snippet_ls[0].dtype

        scales = s_list
        translations = t_list

        A_scaled = [
            (
                reshaped_tensor * s[:, None, None].to(dtype).to(device)
                + t[:, None, None].to(dtype).to(device)
            )
            for reshaped_tensor, s, t in zip(snippet_ls, scales, translations)
        ]

        seq = []
        for i_frame in range(sequence_length):
            tmp = []
            for i_dilation in range(len(A_scaled)):
                tmp.append(A_scaled[i_dilation][indices_list[i_dilation] == i_frame])

            seq.append(torch.cat(tmp).mean(0))

        return torch.cat(seq)[:, None]
