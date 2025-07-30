import argparse
import sys
from pathlib import Path
import shutil
from tqdm import tqdm

def generate_config_content(merged_folder_name):
    """
    Generates the content for the .ini configuration file for the merged dataset.
    Paths are based on the structure seen in the create_orb_slam_input.py script.
    """
    return f"""[RUN]
vocabulary = Vocabulary/ORBvoc.txt
calibration = em/run/settings/calibration_olympusBronchoscope_640x480p.yaml
record = em/run/datasets/phantom/{merged_folder_name}
association = em/run/datasets/phantom/{merged_folder_name}/associations_rd.txt
logs = em/run/datasets/phantom/{merged_folder_name}/logs
patient = false
encoder = false
viewer = true

[PATIENT]
folder = em/run/centerline_frames/phantom

[ENCODER]
sim_encoder = em/run/datasets/phantom/{merged_folder_name}/ca_data.csv

[PRIOR_WEIGHTS]
wT_gba =  10000.0
wT_lba =  1000.0
; wT_moba = 0.0
w_encoder = 1000.0
wx = 10.0
wy = 10.0
wz = 10.0
wroll = 1.0
wpitch = 1.0
wyaw = 1.0

[CBF]
distanceThreshold = 0.05 ; [m]
barrierWeight = 1.0
"""

def merge_slam_data(input_slam_dir: Path, output_merged_dir: Path):
    """
    Merges data from multiple SLAM record directories into a single dataset.
    Adjusts timestamps and filenames to be sequential.
    """
    print(f"--- Starting SLAM Data Merge ---")
    print(f"Input SLAM directory: {input_slam_dir}")
    print(f"Output merged directory: {output_merged_dir}")

    if not input_slam_dir.is_dir():
        print(f"Error: Input SLAM directory not found at {input_slam_dir}", file=sys.stderr)
        sys.exit(1)

    # --- 1. Setup Output Directories ---
    merged_rgb_dir = output_merged_dir / "rgb"
    merged_depth_dir = output_merged_dir / "depth_rd"
    merged_logs_dir = output_merged_dir / "logs" # For consistency with config

    output_merged_dir.mkdir(parents=True, exist_ok=True)
    merged_rgb_dir.mkdir(exist_ok=True)
    merged_depth_dir.mkdir(exist_ok=True)
    merged_logs_dir.mkdir(exist_ok=True)

    # --- 2. Find and Sort Record Directories ---
    # Sort alphabetically; assumes record_videoStem_timestamp format provides reasonable order
    record_dirs = sorted([d for d in input_slam_dir.iterdir() if d.is_dir() and d.name.startswith("record_")])

    if not record_dirs:
        print(f"Error: No 'record_*' subdirectories found in {input_slam_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(record_dirs)} record directories to merge: {[d.name for d in record_dirs]}")

    # --- 3. Process Each Record Directory and Merge Data ---
    all_merged_associations = []
    overall_timestamp_offset = 0.0
    # Default time increment (e.g., 30 FPS) if a video has only one frame or problematic timestamps
    last_valid_time_increment = 1.0 / 30.0
    total_frames_copied = 0

    for record_dir in tqdm(record_dirs, desc="Merging Record Directories"):
        print(f"\nProcessing directory: {record_dir.name}")
        assoc_file_path = record_dir / "associations_rd.txt"
        current_record_rgb_dir = record_dir / "rgb"
        current_record_depth_dir = record_dir / "depth_rd"

        if not assoc_file_path.exists():
            print(f"Warning: Association file not found in {record_dir}. Skipping.", file=sys.stderr)
            continue
        if not current_record_rgb_dir.is_dir() or not current_record_depth_dir.is_dir():
            print(f"Warning: RGB or Depth directory missing in {record_dir}. Skipping.", file=sys.stderr)
            continue

        with open(assoc_file_path, 'r') as f_assoc:
            assoc_lines = [line.strip() for line in f_assoc if line.strip()]

        if not assoc_lines:
            print(f"Warning: Association file in {record_dir} is empty. Skipping.", file=sys.stderr)
            continue

        # Store (original_timestamp_float, original_rgb_filename, original_depth_filename)
        parsed_assoc_data = []
        original_timestamps_in_video = [] # For calculating increment

        for line_idx, line in enumerate(assoc_lines):
            parts = line.split()
            if len(parts) == 4:
                try:
                    orig_ts_float = float(parts[0])
                    original_timestamps_in_video.append(orig_ts_float)
                    parsed_assoc_data.append({
                        "orig_ts": orig_ts_float,
                        "orig_rgb_filename": Path(parts[1]).name, # e.g., "0.000000.png"
                        "orig_depth_filename": Path(parts[3]).name # e.g., "0.000000.png"
                    })
                except ValueError:
                    print(f"Warning: Could not parse timestamp in {assoc_file_path}, line {line_idx+1}: '{line}'. Skipping line.", file=sys.stderr)
                    continue
            else:
                print(f"Warning: Malformed line in {assoc_file_path}, line {line_idx+1}: '{line}'. Skipping line.", file=sys.stderr)
                continue
        
        if not parsed_assoc_data: # No valid lines were parsed
            print(f"Warning: No valid association entries found in {assoc_file_path} for {record_dir.name}. Skipping directory.", file=sys.stderr)
            continue
        
        # Sort by original timestamp to ensure correct processing order within the segment
        parsed_assoc_data.sort(key=lambda x: x["orig_ts"])
        original_timestamps_in_video.sort()


        # Determine time increment for this video segment
        current_video_time_increment = 0.0
        if len(original_timestamps_in_video) > 1:
            increment = original_timestamps_in_video[1] - original_timestamps_in_video[0]
            if increment > 1e-9: # Check for positive, non-negligible increment
                 current_video_time_increment = increment
            else:
                 print(f"Warning: Non-positive or zero time increment ({increment:.6f}) calculated for {record_dir.name}. Using last valid increment: {last_valid_time_increment:.6f}s.", file=sys.stderr)
                 current_video_time_increment = last_valid_time_increment
        else: # Single frame video or only one valid timestamp
            current_video_time_increment = last_valid_time_increment
            print(f"Info: Video {record_dir.name} has <= 1 frame or issues with increment calculation. Using time increment: {current_video_time_increment:.6f}s.")

        frames_in_current_video = 0
        max_orig_ts_in_this_video = 0.0
        if original_timestamps_in_video: # Ensure there's at least one timestamp
            max_orig_ts_in_this_video = original_timestamps_in_video[-1]


        for item_data in tqdm(parsed_assoc_data, desc=f"Frames [{record_dir.stem}]", leave=False):
            orig_ts_float = item_data["orig_ts"]
            orig_rgb_filename = item_data["orig_rgb_filename"]
            orig_depth_filename = item_data["orig_depth_filename"]

            new_global_timestamp = overall_timestamp_offset + orig_ts_float
            new_ts_str = f"{new_global_timestamp:.6f}"

            src_rgb_path = current_record_rgb_dir / orig_rgb_filename
            src_depth_path = current_record_depth_dir / orig_depth_filename

            dest_rgb_path = merged_rgb_dir / f"{new_ts_str}.png"
            dest_depth_path = merged_depth_dir / f"{new_ts_str}.png"

            if not src_rgb_path.exists():
                print(f"Warning: Source RGB file {src_rgb_path} not found. Skipping frame.", file=sys.stderr)
                continue
            if not src_depth_path.exists():
                print(f"Warning: Source Depth file {src_depth_path} not found. Skipping frame.", file=sys.stderr)
                continue

            try:
                shutil.copy2(src_rgb_path, dest_rgb_path) # copy2 preserves metadata
                shutil.copy2(src_depth_path, dest_depth_path)
            except Exception as e:
                print(f"Error copying files for timestamp {orig_ts_float} from {record_dir.name}: {e}. Skipping frame.", file=sys.stderr)
                continue
            
            all_merged_associations.append(f"{new_ts_str} rgb/{new_ts_str}.png {new_ts_str} depth_rd/{new_ts_str}.png")
            frames_in_current_video += 1
            total_frames_copied +=1
            # max_orig_ts_in_this_video was already set to the last timestamp of this video segment

        if frames_in_current_video > 0:
            # Update overall_timestamp_offset for the *next* video.
            # It should start one `current_video_time_increment` after the last frame of *this* video.
            # The start of this video was `overall_timestamp_offset`.
            # The duration of this video's content (relative to its own start) was `max_orig_ts_in_this_video`.
            overall_timestamp_offset += (max_orig_ts_in_this_video + current_video_time_increment)
            last_valid_time_increment = current_video_time_increment
            print(f"Processed {frames_in_current_video} frames from {record_dir.name}. Next video will start at global offset ~{overall_timestamp_offset:.6f}s.")
        else:
            print(f"No frames processed from {record_dir.name}.")


    # --- 4. Write Merged Association File ---
    if not all_merged_associations:
        print("Warning: No associations were merged. Output association file will be empty.", file=sys.stderr)
    
    merged_assoc_file_path = output_merged_dir / "associations_rd.txt"
    print(f"\nWriting {len(all_merged_associations)} entries to merged association file: {merged_assoc_file_path}")
    with open(merged_assoc_file_path, 'w') as f:
        f.write("\n".join(all_merged_associations))
        if all_merged_associations: # Add trailing newline if not empty
            f.write("\n")

    # --- 5. Create Merged Config File ---
    # The config file will refer to the merged data using the name of the output directory itself.
    merged_folder_name_for_config = output_merged_dir.name
    config_content = generate_config_content(merged_folder_name_for_config)
    config_file_path = output_merged_dir / f"config_{merged_folder_name_for_config}.ini"
    
    with open(config_file_path, 'w') as f_config:
        f_config.write(config_content)
    print(f"Created merged config file: {config_file_path}")

    print("\n--- Merge Complete ---")
    print(f"Total frames copied: {total_frames_copied}")
    print(f"Merged data written to: {output_merged_dir}")
    print(f"Merged association file: {merged_assoc_file_path}")
    print(f"Merged config file: {config_file_path}")
    print("Remember to check the generated config file, especially paths and parameters, if your ORB-SLAM setup expects a different base path than 'em/run/datasets/phantom/'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge multiple ORB-SLAM3 preprocessed data folders (output of create_orb_slam_input.py) "
                    "into a single, continuous dataset."
    )
    parser.add_argument(
        "-i", "--input-slam-dir", type=str, required=True,
        help="Path to the root directory from create_orb_slam_input.py. This directory should contain "
             "the 'record_*' subdirectories (e.g., record_video1_timestamp1, record_video2_timestamp2, etc.)."
    )
    parser.add_argument(
        "-o", "--output-merged-dir", type=str, required=True,
        help="Path to the new directory where the merged SLAM dataset will be created. "
             "This directory will contain rgb/, depth_rd/, associations_rd.txt, logs/, and a config_merged.ini file."
    )
    
    args = parser.parse_args()

    input_path = Path(args.input_slam_dir).resolve()
    output_path = Path(args.output_merged_dir).resolve()

    merge_slam_data(input_path, output_path)