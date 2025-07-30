# Author: Bingxin Ke
# Last modified: 2024-11-25

import concurrent.futures
from typing import Union

import matplotlib
import numpy as np
from tqdm import tqdm


def colorize_depth(
    depth: np.ndarray,
    min_depth: float,
    max_depth: float,
    cmap: str = "Spectral_r",
    valid_mask: Union[np.ndarray, None] = None,
) -> np.ndarray:
    assert len(depth.shape) >= 2, "Invalid dimension"

    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1

    if valid_mask is not None:
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    return img_colored_np


def colorize_depth_multi_thread(
    depth: np.ndarray,
    valid_mask: Union[np.ndarray, None] = None,
    chunk_size: int = 4,
    num_threads: int = 4,
    color_map: str = "Spectral",
    verbose: bool = False,
) -> np.ndarray:
    depth = depth.squeeze(1)
    assert 3 == depth.ndim

    n_frame = depth.shape[0]

    if valid_mask is None:
        valid_depth = depth
    else:
        valid_depth = depth[valid_mask]
    min_depth = valid_depth.min()
    max_depth = valid_depth.max()

    def process_chunk(chunk):
        chunk = colorize_depth(
            chunk, min_depth=min_depth, max_depth=max_depth, cmap=color_map
        )
        chunk = (chunk * 255).astype(np.uint8)
        return chunk

    # Pre-allocate the full array
    colored = np.empty((*depth.shape[:3], 3), dtype=np.uint8)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks and store futures with their corresponding indices
        future_to_index = {
            executor.submit(process_chunk, depth[i : i + chunk_size]): i
            for i in range(0, n_frame, chunk_size)
        }

        # Process futures in the order they were submitted
        chunk_iterable = concurrent.futures.as_completed(future_to_index)
        if verbose:
            chunk_iterable = tqdm(
                chunk_iterable,
                desc=" colorizing",
                leave=False,
                total=len(future_to_index),
            )
        for future in chunk_iterable:
            index = future_to_index[future]
            start = index
            end = min(index + chunk_size, n_frame)
            result = future.result()
            colored[start:end] = result
    return colored
