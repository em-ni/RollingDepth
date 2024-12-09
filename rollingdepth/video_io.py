# Copyright 2024 Bingxin Ke, ETH Zurich. All rights reserved.
# Last modified: 2024-12-09
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

import fractions
import logging
from os import PathLike
from typing import List, Optional, Tuple

import av
import einops
import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from tqdm import tqdm


def resize_max_res(
    img: torch.Tensor,
    max_edge_resolution: int,
    resample_method: InterpolationMode = InterpolationMode.BILINEAR,
) -> torch.Tensor:
    """
    Resize image to limit maximum edge length while keeping aspect ratio.

    Args:
        img (`torch.Tensor`):
            Image tensor to be resized. Expected shape: [B, C, H, W]
        max_edge_resolution (`int`):
            Maximum edge length (pixel).
        resample_method (`PIL.Image.Resampling`):
            Resampling method used to resize images.

    Returns:
        `torch.Tensor`: Resized image.
    """
    assert 4 == img.dim(), f"Invalid input shape {img.shape}"

    original_height, original_width = img.shape[-2:]
    downscale_factor = min(
        max_edge_resolution / original_width, max_edge_resolution / original_height
    )

    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)

    resized_img = resize(img, [new_height, new_width], resample_method, antialias=True)
    return resized_img


def load_video_frames(
    input_path: PathLike,
    start_frame: int = 0,
    frame_count: int = 0,
    processing_res: int = 0,
    resample_method: str = "BILINEAR",
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Size]:
    assert start_frame >= 0

    # Open the video file
    container = av.open(input_path)
    stream = container.streams.video[0]

    # Calculate end frame
    end_before = start_frame + frame_count if frame_count > 0 else np.inf

    # Set stream to decode only frames we need
    stream.thread_type = "AUTO"  # Enable multithreading

    # Iterate through frames
    if verbose:
        frame_iterable = tqdm(
            container.decode(stream),  # type: ignore
            desc="Loading frames",
            leave=False,
        )
    else:
        frame_iterable = container.decode(stream)  # type: ignore
    frame_ls = []
    original_res: torch.Size = None  # type: ignore
    for i, frame in enumerate(frame_iterable):
        if i >= start_frame and i < end_before:
            # Convert frame to numpy array and then to torch tensor
            frame_array = frame.to_ndarray(format="rgb24")
            frame = torch.from_numpy(frame_array.copy()).float()

            # original resolution before resizing
            if original_res is None:
                original_res = frame.shape[:2]

            frame = einops.rearrange(frame, "h w c -> 1 c h w")

            # Resize if requested
            if processing_res > 0:
                frame = resize_max_res(
                    frame,
                    max_edge_resolution=processing_res,
                    resample_method=InterpolationMode.__getitem__(resample_method),
                )

            # Normalize to to [-1, 1]
            frame_norm = (frame / 255.0) * 2.0 - 1.0

            frame_ls.append(frame_norm)

        if i >= end_before:
            break

    container.close()

    if 0 == len(frame_ls):
        raise RuntimeError(f"No frame is loaded from {input_path}")

    frames = torch.cat(frame_ls, dim=0)  # [N C H W]

    return frames, original_res


def write_video_from_numpy(
    frames: np.ndarray,  # shape [n h w 3]
    output_path: PathLike,
    fps: float = 30.0,
    codec: Optional[str] = None,  # Let PyAV choose default codec
    crf: int = 23,
    preset: str = "medium",
    verbose: bool = False,
) -> None:
    if len(frames.shape) != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected shape [n, height, width, 3], got {frames.shape}")
    if frames.dtype != np.uint8:
        raise ValueError(f"Expected dtype uint8, got {frames.dtype}")

    n_frames, height, width, _ = frames.shape

    # Try to determine codec from output format if not specified
    if codec is None:
        codecs_to_try = ["libx264", "h264", "mpeg4", "mjpeg"]
    else:
        codecs_to_try = [codec]

    fps_rational = fractions.Fraction(fps).limit_denominator()

    # Try available codecs
    for try_codec in codecs_to_try:
        try:
            container = av.open(output_path, mode="w")
            stream = container.add_stream(try_codec, rate=fps_rational)
            if verbose:
                logging.info(f"Using codec: {try_codec}")
            break
        except av.codec.codec.UnknownCodecError:  # type: ignore
            if try_codec == codecs_to_try[-1]:  # Last codec in list
                raise ValueError(
                    f"No working codec found. Tried: {codecs_to_try}. "
                    "Please install ffmpeg with necessary codecs."
                )
            continue

    stream.width = width  # type: ignore
    stream.height = height  # type: ignore
    stream.pix_fmt = "yuv420p"  # type: ignore

    # Only set these options for x264-compatible codecs
    if try_codec in ["libx264", "h264"]:  # type: ignore
        stream.options = {"crf": str(crf), "preset": preset}  # type: ignore

    # Create a single VideoFrame object and reuse it
    video_frame = av.VideoFrame(width, height, "rgb24")

    frames_iterable = range(n_frames)
    if verbose:
        frames_iterable = tqdm(frames_iterable, desc="Writing video", total=n_frames)

    try:
        for frame_idx in frames_iterable:
            # Get view of current frame
            current_frame = frames[frame_idx]

            # Update frame data in-place
            video_frame.to_ndarray()[:] = current_frame

            packet = stream.encode(video_frame)  # type: ignore
            container.mux(packet)  # type: ignore

        # Flush the stream
        packet = stream.encode(None)  # type: ignore
        container.mux(packet)  # type: ignore
    finally:
        container.close()  # type: ignore


def get_video_fps(video_path: PathLike) -> float:
    # Open the video file
    container = av.open(video_path)

    # Get the video stream
    video_stream = container.streams.video[0]

    # Calculate FPS from the stream's time base and average frame rate
    fps = float(video_stream.average_rate)  # type: ignore

    # Close the container
    container.close()

    return fps


def concatenate_videos_horizontally_torch(
    video1: torch.Tensor,
    video2: torch.Tensor,
    gap: int = 0,
    gap_color: Optional[List[int]] = None,
):
    # Convert to torch tensors if they aren't already
    if isinstance(video1, np.ndarray):
        video1 = torch.from_numpy(video1)  # [N, 3, H, W]
    if isinstance(video2, np.ndarray):
        video2 = torch.from_numpy(video2)  # [N, 3, H, W]

    # Get target size
    N, C, H1, W1 = video1.shape

    # Resize video2 to match height of video1
    video2_resized = resize(video2, [H1, W1], antialias=True)

    if gap > 0:
        # Create gap tensor
        if gap_color is None:
            gap_color = [0, 0, 0]  # Default to black

        gap_tensor = torch.ones(
            (N, C, H1, gap),
            dtype=video1.dtype,
            device=video1.device,
        ) * torch.tensor(gap_color).int().view(3, 1, 1)

        # Concatenate with gap
        concatenated = torch.cat([video1, gap_tensor, video2_resized], dim=3)
    else:
        # Concatenate without gap
        concatenated = torch.cat([video1, video2_resized], dim=3)

    # Concatenate along width dimension
    concatenated = torch.cat([video1, video2_resized], dim=3)

    return concatenated
