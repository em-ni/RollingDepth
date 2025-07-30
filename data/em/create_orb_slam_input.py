import argparse
import os
import sys
from pathlib import Path
import numpy as np
import cv2  # OpenCV for image saving and video reading
from PIL import Image # Pillow for saving 16-bit PNG
from tqdm import tqdm
import av # PyAV for potentially more robust video reading / FPS info
import time # To generate a base timestamp if needed

def get_video_fps(video_path):
    """Gets FPS using PyAV, falling back to OpenCV."""
    fps = 0
    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        container.close()
        if fps > 0:
            # print(f"Determined FPS using PyAV: {fps:.3f}") # Less verbose in batch mode
            return fps
        else:
            print(f"Warning [{video_path.name}]: PyAV reported FPS <= 0. Falling back to OpenCV.", file=sys.stderr)
    except Exception as e:
        print(f"Warning [{video_path.name}]: Error getting FPS with PyAV: {e}", file=sys.stderr)
        print(f"Info [{video_path.name}]: Falling back to OpenCV for FPS.", file=sys.stderr)

    # Fallback to OpenCV
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file (OpenCV fallback): {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
         raise ValueError(f"Could not determine FPS for {video_path} using PyAV or OpenCV.")
    # print(f"Determined FPS using OpenCV fallback: {fps:.3f}") # Less verbose in batch mode
    return fps

def process_video_for_slam(video_path, rolling_depth_output_dir, slam_data_dir, depth_scale_factor=5000.0):
    """
    Processes a single video and RollingDepth output to create ORB-SLAM3 compatible input.
    """
    print(f"--- Processing Video: {video_path.name} ---")
    print(f"RollingDepth output dir: {rolling_depth_output_dir}")
    print(f"SLAM data output dir: {slam_data_dir}")
    print(f"Using Depth Scale Factor: {depth_scale_factor}")

    # --- 1. Setup Directories ---
    rgb_out_dir = slam_data_dir / f"rgb"
    depth_out_dir = slam_data_dir / f"depth_rd"
    rgb_out_dir.mkdir(parents=True, exist_ok=True) # Create specific rgb dir
    depth_out_dir.mkdir(parents=True, exist_ok=True) # Create specific depth dir
    association_file_path = slam_data_dir / f"associations_rd.txt"

    # --- 2. Load Depth Data ---
    # Expecting depth file named like the video stem + _pred.npy
    depth_npy_file = rolling_depth_output_dir / f"{video_path.stem}_pred.npy"
    if not depth_npy_file.exists():
        # Try alternative naming convention if default not found (e.g. if run_video output name differed)
        potential_files = list(rolling_depth_output_dir.glob(f'{video_path.stem}*_pred.npy'))
        if potential_files:
             depth_npy_file = potential_files[0]
             print(f"Info [{video_path.name}]: Default depth file name not found. Using: {depth_npy_file.name}")
        else:
             # More specific error for the video being processed
             raise FileNotFoundError(f"Depth file '{video_path.stem}_pred.npy' (or similar) not found in {rolling_depth_output_dir}")

    print(f"Loading depth data from {depth_npy_file}...")
    try:
        depth_data = np.load(depth_npy_file) # Shape: (N, H, W)
    except ValueError as e:
         print(f"Error loading NumPy file {depth_npy_file}. It might be corrupted or empty: {e}", file=sys.stderr)
         raise # Re-raise exception to be caught by the main loop

    if depth_data.ndim != 3 or depth_data.shape[0] == 0:
        raise ValueError(f"Loaded depth data has unexpected shape {depth_data.shape} or is empty.")

    num_depth_frames = depth_data.shape[0]
    depth_h, depth_w = depth_data.shape[1], depth_data.shape[2]
    print(f"Found {num_depth_frames} depth frames (Resolution: {depth_w}x{depth_h}).")

    # --- 3. Get Video FPS ---
    try:
        fps = get_video_fps(video_path)
        print(f"Video FPS: {fps:.3f}")
    except Exception as e:
        print(f"Error getting video FPS: {e}. Cannot proceed with this video.", file=sys.stderr)
        raise # Re-raise exception

    # --- 4. Process Frames and Generate Timestamps ---
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    associations = []
    time_increment = 1.0 / fps
    base_timestamp = 0.0 # Use 0.0 start time for simplicity and relative SLAM

    # Hard-coded camera parameters as requested
    near_plane = 0.1  # in meters
    far_plane = 1000.0  # in meters

    print(f"Processing {num_depth_frames} frames based on depth data...")
    print(f"Using near_plane={near_plane}, far_plane={far_plane}, depth_map_factor={depth_scale_factor}")
    
    frames_processed = 0
    # Use TQDM specific to this video's frames
    for frame_idx in tqdm(range(num_depth_frames), desc=f"Frames [{video_path.stem}]", leave=False):
        ret, frame_rgb = cap.read()
        if not ret:
            print(f"\nWarning [{video_path.name}]: Could not read RGB frame {frame_idx} from video. Stopping early for this video.", file=sys.stderr)
            break

        timestamp = base_timestamp + frame_idx * time_increment
        timestamp_str = f"{timestamp:.6f}"

        # --- Save RGB ---
        rgb_filename = f"{timestamp_str}.png"
        rgb_filepath = rgb_out_dir / rgb_filename
        if frame_idx == 0: # Check resolution only once
            rgb_h, rgb_w, _ = frame_rgb.shape
            if rgb_h != depth_h or rgb_w != depth_w:
                 print(f"Warning [{video_path.name}]: RGB resolution ({rgb_w}x{rgb_h}) differs from Depth ({depth_w}x{depth_h}).", file=sys.stderr)
        cv2.imwrite(str(rgb_filepath), frame_rgb)

        # --- Process and save Depth (UPDATED) ---
        depth_frame = depth_data[frame_idx]

        # Step 1: Normalize depth values to 0.0-1.0 range
        # The raw data is roughly in -1 to 1 range.
        depth_normalized = (depth_frame + 1.0) / 2.0

        # Ensure values are within the initial 0-1 range
        depth_normalized = np.clip(depth_normalized, 0.0, 1.0)

        # >>> FIX: Invert the normalized depth <<<
        # This is the crucial step. It maps:
        # - close objects (raw ~+1 -> normalized ~1.0) to a new value of ~0.0
        # - far objects   (raw ~-1 -> normalized ~0.0) to a new value of ~1.0
        # This aligns the data with what the linearization formula expects.
        depth_normalized = 1.0 - depth_normalized

        # Step 2: Linearize the depth exactly as in the reference code
        depth_linear = (2.0 * near_plane * far_plane) / (
            far_plane + near_plane - (2.0 * depth_normalized - 1.0) * (far_plane - near_plane)
        )
        
        # Step 3: Apply depth map factor and convert to uint16 format
        depth_scaled = (depth_linear * depth_scale_factor).astype(np.uint16)
        
        depth_filename = f"{timestamp_str}.png"
        depth_filepath = depth_out_dir / depth_filename
        try:
            img_depth_pil = Image.fromarray(depth_scaled)
            img_depth_pil.save(str(depth_filepath))
        except Exception as e:
             print(f"\nError saving depth frame {frame_idx} ({depth_filepath}): {e}. Skipping frame.", file=sys.stderr)
             continue # Skip association for this frame if saving failed

        # --- Add to associations list ---
        associations.append(f"{timestamp_str} rgb/{rgb_filename} {timestamp_str} depth_rd/{depth_filename}")
        frames_processed += 1

    cap.release()
    print(f"-> Successfully processed and saved {frames_processed} frame pairs for {video_path.name}.")

    if frames_processed == 0:
         print(f"Error: No frames were successfully processed for {video_path.name}. Check video file and depth data.", file=sys.stderr)
         # No point writing an empty association file
         return # Exit this function call

    if frames_processed < num_depth_frames:
         print(f"Warning [{video_path.name}]: Only processed {frames_processed} frames, but found {num_depth_frames} depth frames. Video might be shorter or reading stopped.", file=sys.stderr)

    # --- 5. Write Association File ---
    print(f"Writing {len(associations)} entries to association file: {association_file_path}")
    with open(association_file_path, 'w') as f:
        f.write("\n".join(associations))
        f.write("\n")
    print(f"--- Finished Video: {video_path.name} ---")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare RollingDepth output for ORB-SLAM3 (TUM RGB-D format) for multiple videos."
    )
    parser.add_argument(
        "-i", "--input-video-dir", type=str, required=True,
        help="Path to the directory containing the original input video files (e.g., *.mp4, *.avi)."
    )
    parser.add_argument(
        "-d", "--depth-dir", type=str, required=True,
        help="Path to the directory containing ALL RollingDepth output *_pred.npy files."
    )
    parser.add_argument(
        "-o", "--output-slam-dir", type=str, required=True,
        help="Path to the main directory where SLAM-ready subdirectories will be created (one per video)."
    )
    parser.add_argument(
        "--depth-scale", type=float, default=5000.0,
        help="Scale factor for depth values (e.g., 5000 for TUM default). MUST match ORB-SLAM3 YAML's DepthMapFactor."
    )

    args = parser.parse_args()

    input_dir = Path(args.input_video_dir)
    depth_dir = Path(args.depth_dir)
    output_root_dir = Path(args.output_slam_dir)

    if not input_dir.is_dir():
        print(f"Error: Input video directory not found at {input_dir}", file=sys.stderr)
        sys.exit(1)
    if not depth_dir.is_dir():
         print(f"Error: RollingDepth output directory not found at {depth_dir}", file=sys.stderr)
         sys.exit(1)

    # Create the main output directory if it doesn't exist
    output_root_dir.mkdir(parents=True, exist_ok=True)

    # Create the configs directory
    configs_dir = output_root_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    # --- Find video files ---
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv'] # Add more if needed
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(input_dir.glob(f'*{ext}')))
        video_files.extend(list(input_dir.glob(f'*{ext.upper()}'))) # Also check uppercase

    if not video_files:
        print(f"Error: No video files found in {input_dir} with extensions {video_extensions}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(video_files)} potential video files in {input_dir}.")

    # --- Process each video ---
    processed_count = 0
    failed_count = 0
    for video_file in video_files:
        # Create specific output directory for this video
        unix_timestamp = int(time.time())  # Current Unix timestamp
        video_stem = video_file.stem # This is NAMEOFTHEVIDEO
        video_output_folder_name = f"record_{video_stem}_{unix_timestamp}"  # This is NAMEOFTHEVIDEOFOLDER
        video_output_subdir = output_root_dir / video_output_folder_name
        video_output_subdir.mkdir(parents=True, exist_ok=True)

        try:
            process_video_for_slam(
                video_path=video_file,
                rolling_depth_output_dir=depth_dir,
                slam_data_dir=video_output_subdir,
                depth_scale_factor=args.depth_scale
            )

            # --- Create .ini config file ---
            config_file_name = f"config_{video_stem}.ini" # Uses NAMEOFTHEVIDEO
            config_file_path = configs_dir / config_file_name
            
            config_content = f"""[RUN]
vocabulary = Vocabulary/ORBvoc.txt
calibration = em/run/settings/calibration_olympusBronchoscope_1280x720p.yaml
record = em/run/datasets/phantom/{video_output_folder_name}
association = em/run/datasets/phantom/{video_output_folder_name}/associations_rd.txt
logs = em/run/datasets/phantom/{video_output_folder_name}/logs
patient = false
encoder = false
viewer = false

[PATIENT]
folder = em/run/centerline_frames/phantom

[ENCODER]
sim_encoder = em/run/datasets/phantom/{video_output_folder_name}/ca_data.csv
"""
            with open(config_file_path, 'w') as f_config:
                f_config.write(config_content)
            print(f"Created config file: {config_file_path}")
            # --- End of .ini config file creation ---

            processed_count += 1
        except FileNotFoundError as e:
            print(f"Error processing {video_file.name}: Required file not found - {e}. Skipping.", file=sys.stderr)
            failed_count += 1
        except Exception as e:
            print(f"!!!--- Critical Error processing {video_file.name}: {e} ---!!! Skipping.", file=sys.stderr)
            # import traceback # Uncomment for detailed debugging
            # traceback.print_exc() # Uncomment for detailed debugging
            failed_count += 1
        print("-" * 50) # Separator

    # --- Final Summary ---
    print("\n======= Batch Processing Summary =======")
    print(f"Total videos found: {len(video_files)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed/Skipped: {failed_count}")
    print(f"SLAM-ready data created in subdirectories under: {output_root_dir}")
    print(f"Configuration files created in: {configs_dir}")
    print(f"Remember to use DepthMapFactor = {args.depth_scale} in ORB-SLAM3 YAML configuration.")
    print("========================================")