#!/usr/bin/env python3
"""Minimal evaluation script for RollingDepth on dataset"""

import argparse
import logging
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm

from rollingdepth import RollingDepthPipeline

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def create_video_from_frames(rgb_dir, start_idx, num_frames, output_path):
    """Create a temporary video from RGB frames"""
    rgb_files = sorted(rgb_dir.glob("*.png"))
    
    frame = cv2.imread(str(rgb_files[0]))
    h, w = frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, 15.0, (w, h))
    
    for i in range(start_idx, min(start_idx + num_frames, len(rgb_files))):
        frame = cv2.imread(str(rgb_files[i]))
        out.write(frame)
    
    out.release()


def _fit_affine_scale_shift(x: np.ndarray, y: np.ndarray, eps: float = 1e-8):
    """Fit y ~ a * x + b via least squares (closed-form) on 1D arrays.
    Returns (a, b). If var(x) is too small, returns (0, mean(y))."""
    x_mean = x.mean()
    y_mean = y.mean()
    var_x = np.mean((x - x_mean) ** 2)
    if var_x < eps or not np.isfinite(var_x):
        return 0.0, float(y_mean)
    cov_xy = np.mean((x - x_mean) * (y - y_mean))
    a = cov_xy / var_x
    b = y_mean - a * x_mean
    if not np.isfinite(a):
        a = 0.0
    if not np.isfinite(b):
        b = float(y_mean)
    return float(a), float(b)


def load_gt_depth(depth_dir, start_idx, num_frames, depth_map_factor=5000.0):
    """Load ground truth depth frames (already in meters from simulator)"""
    depth_files = sorted(depth_dir.glob("*.png"))
    depths = []
    
    for i in range(start_idx, min(start_idx + num_frames, len(depth_files))):
        depth_img = Image.open(depth_files[i])
        depth_uint16 = np.array(depth_img).astype(np.float32)
        depth_meters = depth_uint16 / depth_map_factor
        depths.append(depth_meters)
    
    return np.array(depths)


def evaluate_dataset(dataset_dir, checkpoint="prs-eth/rollingdepth-v1-0", depth_map_factor: float = 5000.0):
    """Evaluate model on dataset"""
    dataset_path = Path(dataset_dir)
    video_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir() and (d / "rgb").exists()])
    
    logger.info(f"Found {len(video_dirs)} videos")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = RollingDepthPipeline.from_pretrained(checkpoint, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    
    # Evaluation parameters
    fps = 15
    chunk_duration = 30  
    chunk_frames = fps * chunk_duration  

    all_mae = []
    all_rmse = []
    
    for video_dir in tqdm(video_dirs, desc="Videos", unit="video"):
        rgb_dir = video_dir / "rgb"
        depth_dir = video_dir / "depth"
        pred_dir = video_dir / "pred"
        pred_dir.mkdir(exist_ok=True)  # Create pred folder for this video
        
        num_rgb = len(list(rgb_dir.glob("*.png")))
        num_depth = len(list(depth_dir.glob("*.png")))
        
        tqdm.write(f"{video_dir.name}: {num_rgb} RGB, {num_depth} depth")
        
        # Process chunks
        num_chunks = max(1, num_rgb // chunk_frames)
        
        for chunk_idx in tqdm(range(num_chunks), desc=f"  Chunks", leave=False, unit="chunk"):
            start_idx = chunk_idx * chunk_frames
            
            # Create temp video
            temp_video = Path(f"/tmp/chunk_{chunk_idx}.mp4")
            create_video_from_frames(rgb_dir, start_idx, chunk_frames, temp_video)
            
            # Run inference
            with torch.no_grad():
                result = pipe(
                    input_video_path=str(temp_video),
                    processing_res=768,
                    dilations=[1, 25],
                    cap_dilation=True,
                    snippet_lengths=[3],
                    refine_step=0,
                )
            
            pred_raw = result.depth_pred.squeeze(1).float().cpu().numpy()  # [N, H, W], values ~[-1, 1]
            
            # Load GT
            gt_depth = load_gt_depth(depth_dir, start_idx, chunk_frames, depth_map_factor=depth_map_factor)  # [N, H_gt, W_gt] in meters
            
            # Align sizes - match both frame count and spatial resolution
            min_frames = min(len(pred_raw), len(gt_depth))
            pred_raw = pred_raw[:min_frames]
            gt_depth = gt_depth[:min_frames]
            
            # Resize predictions to match GT spatial resolution (for evaluation and saving)
            if pred_raw.shape[1:] != gt_depth.shape[1:]:
                gt_h, gt_w = gt_depth.shape[1:]
                pred_resized = np.zeros_like(gt_depth, dtype=np.float32)
                for i in range(min_frames):
                    pred_resized[i] = cv2.resize(
                        pred_raw[i].astype(np.float32), (gt_w, gt_h), interpolation=cv2.INTER_LINEAR
                    )
            else:
                pred_resized = pred_raw.astype(np.float32)

            # Align predictions to GT in inverse depth space per-frame, then convert to meters
            aligned_pred_depth_m = np.zeros_like(gt_depth, dtype=np.float32)
            depth_files = sorted(depth_dir.glob("*.png"))
            for i in range(min_frames):
                gt_frame = gt_depth[i]
                pred_frame = pred_resized[i]
                valid_mask = np.isfinite(gt_frame) & (gt_frame > 0.0)
                if valid_mask.sum() == 0:
                    # Nothing valid; leave zeros
                    continue
                x = pred_frame[valid_mask].astype(np.float32).ravel()
                y = (1.0 / np.clip(gt_frame[valid_mask], 1e-6, None)).astype(np.float32).ravel()  # inverse GT
                a, b = _fit_affine_scale_shift(x, y)
                inv_pred = a * pred_frame + b  # full-frame inverse depth
                inv_pred = np.clip(inv_pred, 1e-6, 1e6)
                depth_m = 1.0 / inv_pred
                aligned_pred_depth_m[i] = depth_m.astype(np.float32)
            
            # Compute metrics (only valid pixels on GT)
            valid_mask = (gt_depth > 0.0) & np.isfinite(gt_depth)
            if valid_mask.sum() > 0:
                diff = (aligned_pred_depth_m - gt_depth)[valid_mask]
                mae = np.abs(diff).mean()
                rmse = np.sqrt((diff ** 2).mean())
                all_mae.append(float(mae))
                all_rmse.append(float(rmse))
                tqdm.write(f"    Chunk {chunk_idx}: MAE={mae:.4f}m, RMSE={rmse:.4f}m")

            # Save aligned predicted depth frames as uint16 PNG using GT filenames
            for frame_in_chunk, frame_in_dataset in enumerate(range(start_idx, start_idx + min_frames)):
                if frame_in_dataset < len(depth_files):
                    orig_depth_file = depth_files[frame_in_dataset]
                    pred_filename = orig_depth_file.name
                    pred_depth_uint16 = (np.clip(aligned_pred_depth_m[frame_in_chunk], 0.0, 1000.0) * depth_map_factor).astype(np.uint16)
                    pred_filepath = pred_dir / pred_filename
                    pred_img = Image.fromarray(pred_depth_uint16)
                    pred_img.save(str(pred_filepath))
            
            temp_video.unlink()
    
    # Summary
    if all_mae:
        tqdm.write(f"\n{'='*50}")
        tqdm.write(f"Evaluated {len(all_mae)} chunks")
        tqdm.write(f"Overall MAE:  {np.mean(all_mae):.4f}m (±{np.std(all_mae):.4f})")
        tqdm.write(f"Overall RMSE: {np.mean(all_rmse):.4f}m (±{np.std(all_rmse):.4f})")
        tqdm.write(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str, help="Path to dataset directory")
    parser.add_argument("--checkpoint", default="prs-eth/rollingdepth-v1-0", help="Model checkpoint")
    parser.add_argument("--depth-map-factor", type=float, default=5000.0, help="Depth scaling factor used by the dataset (e.g., 5000 for TUM)")
    args = parser.parse_args()
    
    evaluate_dataset(args.dataset_dir, args.checkpoint, args.depth_map_factor)
