# Utility script for preparing datasets for RollingDepth finetuning

import argparse
import logging
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def organize_frames_from_videos(
    input_dir: str,
    output_dir: str,
    depth_source: str = 'depth_maps',
    max_frames_per_video: int = 0,
):
    """
    Organize RGB and depth frames from video files into structured directory.
    
    Args:
        input_dir: Directory containing videos and depth data
        output_dir: Output directory for organized frames
        depth_source: Subdirectory name containing depth maps
        max_frames_per_video: Max frames to extract per video (0 = all)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find video files
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    video_files = [f for f in input_path.glob('*') if f.suffix.lower() in video_extensions]
    
    logger.info(f"Found {len(video_files)} video files")
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        video_name = video_file.stem
        
        # Create output directories
        video_output = output_path / video_name
        rgb_output = video_output / 'rgb'
        depth_output = video_output / 'depth'
        
        rgb_output.mkdir(parents=True, exist_ok=True)
        depth_output.mkdir(parents=True, exist_ok=True)
        
        # Extract frames from video
        cap = cv2.VideoCapture(str(video_file))
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames_per_video > 0 and frame_count >= max_frames_per_video:
                break
            
            # Save frame
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_path = rgb_output / f"frame_{frame_count:06d}.png"
            Image.fromarray(frame_bgr).save(str(frame_path))
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {frame_count} frames from {video_name}")
        
        # Copy corresponding depth maps if they exist
        depth_src_dir = input_path / depth_source / video_name
        if depth_src_dir.exists():
            depth_files = sorted(depth_src_dir.glob('*.npy')) + sorted(depth_src_dir.glob('*.png'))
            for i, depth_file in enumerate(depth_files[:frame_count]):
                if depth_file.suffix == '.npy':
                    # Copy numpy file
                    dest = depth_output / f"frame_{i:06d}.npy"
                    np.save(str(dest), np.load(str(depth_file)))
                else:
                    # Copy image file
                    dest = depth_output / f"frame_{i:06d}.png"
                    Image.open(str(depth_file)).save(str(dest))


def create_depth_from_disparity(
    disparity_dir: str,
    output_dir: str,
    baseline: float = 1.0,
    focal_length: float = 1.0,
):
    """
    Convert disparity maps to depth maps using stereo formula.
    
    Args:
        disparity_dir: Directory containing disparity maps
        output_dir: Output directory for depth maps
        baseline: Baseline distance between cameras (meters)
        focal_length: Focal length in pixels
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    disparity_files = sorted(Path(disparity_dir).glob('*.npy')) + \
                     sorted(Path(disparity_dir).glob('*.png'))
    
    logger.info(f"Converting {len(disparity_files)} disparity maps to depth")
    
    for disp_file in tqdm(disparity_files):
        # Load disparity
        if disp_file.suffix == '.npy':
            disparity = np.load(str(disp_file))
        else:
            disparity = np.array(Image.open(str(disp_file)), dtype=np.float32)
        
        # Convert to depth using stereo formula: depth = (baseline * focal_length) / disparity
        depth = np.zeros_like(disparity)
        valid_mask = disparity > 0
        depth[valid_mask] = (baseline * focal_length) / disparity[valid_mask]
        
        # Save
        output_file = output_path / f"{disp_file.stem}_depth.npy"
        np.save(str(output_file), depth)
    
    logger.info(f"Converted disparity maps saved to {output_dir}")


def normalize_depth_dataset(
    dataset_dir: str,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
    in_place: bool = False,
):
    """
    Normalize all depth maps in dataset to [-1, 1] range.
    
    Args:
        dataset_dir: Dataset directory with structure: video_*/depth/*
        min_depth: Minimum depth value
        max_depth: Maximum depth value
        in_place: If True, overwrite original files. Otherwise save to _normalized
    """
    dataset_path = Path(dataset_dir)
    video_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    
    total_frames = 0
    for video_dir in video_dirs:
        depth_dir = video_dir / 'depth'
        if not depth_dir.exists():
            continue
        
        depth_files = sorted(depth_dir.glob('*.npy')) + sorted(depth_dir.glob('*.png'))
        
        for depth_file in tqdm(depth_files, desc=f"Normalizing {video_dir.name}"):
            # Load depth
            if depth_file.suffix == '.npy':
                depth = np.load(str(depth_file))
            else:
                depth = np.array(Image.open(str(depth_file)), dtype=np.float32)
                if depth.max() > 255:
                    depth = depth / 65535.0
                else:
                    depth = depth / 255.0
            
            # Normalize
            depth = np.clip(depth, min_depth, max_depth)
            depth = (depth - min_depth) / (max_depth - min_depth)  # [0, 1]
            depth = depth * 2.0 - 1.0  # [-1, 1]
            
            # Save
            if in_place:
                output_file = depth_file.with_suffix('.npy')
            else:
                output_file = depth_file.with_stem(depth_file.stem + '_normalized').with_suffix('.npy')
            
            np.save(str(output_file), depth)
            total_frames += 1
    
    logger.info(f"Normalized {total_frames} depth frames")


def split_dataset(
    dataset_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    """
    Split dataset into train/val/test subsets.
    
    Args:
        dataset_dir: Dataset directory with structure: video_*/
        output_dir: Output directory for split datasets
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    dataset_path = Path(dataset_dir)
    video_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create split directories
    splits = {
        'train': output_path / 'train',
        'val': output_path / 'val',
        'test': output_path / 'test',
    }
    for split_dir in splits.values():
        split_dir.mkdir(exist_ok=True)
    
    # Calculate split indices
    num_videos = len(video_dirs)
    train_count = int(num_videos * train_ratio)
    val_count = int(num_videos * val_ratio)
    
    split_assignment = ['train'] * train_count + ['val'] * val_count + ['test'] * (num_videos - train_count - val_count)
    
    # Copy videos to splits
    split_info = {}
    for video_dir, split in zip(video_dirs, split_assignment):
        dest_dir = splits[split] / video_dir.name
        
        # Copy directory tree
        import shutil
        if dest_dir.exists():
            shutil.rmtree(str(dest_dir))
        shutil.copytree(str(video_dir), str(dest_dir))
        
        if split not in split_info:
            split_info[split] = []
        split_info[split].append(video_dir.name)
    
    # Save split info
    info_file = output_path / 'split_info.json'
    with open(str(info_file), 'w') as f:
        json.dump(split_info, f, indent=2)
    
    logger.info(f"Dataset split saved to {output_dir}")
    for split, count in [('train', len(split_info['train'])), 
                         ('val', len(split_info['val'])), 
                         ('test', len(split_info['test']))]:
        logger.info(f"  {split}: {count} videos")


def validate_dataset(dataset_dir: str) -> bool:
    """
    Validate dataset structure and file integrity.
    
    Args:
        dataset_dir: Dataset directory to validate
    
    Returns:
        True if valid, False otherwise
    """
    dataset_path = Path(dataset_dir)
    video_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    
    if not video_dirs:
        logger.error("No video directories found")
        return False
    
    issues = []
    
    for video_dir in video_dirs:
        rgb_dir = video_dir / 'rgb'
        depth_dir = video_dir / 'depth'
        
        if not rgb_dir.exists() or not depth_dir.exists():
            issues.append(f"{video_dir.name}: Missing rgb or depth directory")
            continue
        
        rgb_files = sorted(rgb_dir.glob('*.png')) + sorted(rgb_dir.glob('*.jpg'))
        depth_files = sorted(depth_dir.glob('*.npy')) + sorted(depth_dir.glob('*.png'))
        
        if len(rgb_files) == 0:
            issues.append(f"{video_dir.name}: No RGB frames found")
        
        if len(depth_files) == 0:
            issues.append(f"{video_dir.name}: No depth frames found")
        
        if len(rgb_files) != len(depth_files):
            issues.append(
                f"{video_dir.name}: RGB/depth mismatch "
                f"({len(rgb_files)} RGB vs {len(depth_files)} depth)"
            )
    
    if issues:
        logger.error("Dataset validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    else:
        logger.info(f"Dataset validation successful: {len(video_dirs)} videos with valid structure")
        return True


def main():
    parser = argparse.ArgumentParser(description="Utility for preparing RollingDepth datasets")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # organize command
    organize_parser = subparsers.add_parser('organize', help='Organize frames from videos')
    organize_parser.add_argument('--input_dir', required=True)
    organize_parser.add_argument('--output_dir', required=True)
    organize_parser.add_argument('--max_frames_per_video', type=int, default=0)
    
    # normalize command
    normalize_parser = subparsers.add_parser('normalize', help='Normalize depth maps')
    normalize_parser.add_argument('--dataset_dir', required=True)
    normalize_parser.add_argument('--min_depth', type=float, default=0.1)
    normalize_parser.add_argument('--max_depth', type=float, default=10.0)
    normalize_parser.add_argument('--in_place', action='store_true')
    
    # split command
    split_parser = subparsers.add_parser('split', help='Split dataset into train/val/test')
    split_parser.add_argument('--dataset_dir', required=True)
    split_parser.add_argument('--output_dir', required=True)
    split_parser.add_argument('--train_ratio', type=float, default=0.8)
    split_parser.add_argument('--val_ratio', type=float, default=0.1)
    
    # validate command
    validate_parser = subparsers.add_parser('validate', help='Validate dataset')
    validate_parser.add_argument('--dataset_dir', required=True)
    
    args = parser.parse_args()
    
    if args.command == 'organize':
        organize_frames_from_videos(args.input_dir, args.output_dir, args.max_frames_per_video)
    elif args.command == 'normalize':
        normalize_depth_dataset(args.dataset_dir, args.min_depth, args.max_depth, args.in_place)
    elif args.command == 'split':
        split_dataset(args.dataset_dir, args.output_dir, args.train_ratio, args.val_ratio)
    elif args.command == 'validate':
        validate_dataset(args.dataset_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
