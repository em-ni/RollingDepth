import os
import shutil
import re

def organize_ground_truth_files():
    """
    Organizes ground truth files for video sequences.

    For each folder in 'slam/bronchoscopy_dataset' starting with 'record' and
    ending with a timestamp, this script creates a 'gt' subfolder. It then
    copies the corresponding ground truth .txt file from 'bronchoscopy_dataset/gt'
    into this 'gt' subfolder and renames it to 'gt_wTc.txt'.
    """
    slam_dataset_base_path = "slam/bronchoscopy_dataset"
    gt_source_base_path = "bronchoscopy_dataset/gt"

    # Check if base paths exist
    if not os.path.isdir(slam_dataset_base_path):
        print(f"Error: Slam dataset path '{slam_dataset_base_path}' does not exist.")
        return
    if not os.path.isdir(gt_source_base_path):
        print(f"Error: Ground truth source path '{gt_source_base_path}' does not exist.")
        return

    # Regex to match folder names like 'record_real_seq_000_part_1_dif_1_1746680589'
    # It captures:
    # group(1): the part between 'record_' and the final '_timestamp' (e.g., 'real_seq_000_part_1_dif_1')
    # group(2): the timestamp (e.g., '1746680589')
    folder_pattern = re.compile(r"^record_(.*)_(\d+)$")

    print(f"Scanning '{slam_dataset_base_path}' for record folders...")

    for item_name in os.listdir(slam_dataset_base_path):
        item_path = os.path.join(slam_dataset_base_path, item_name)

        if os.path.isdir(item_path):
            match = folder_pattern.match(item_name)
            if match:
                print(f"\nProcessing folder: {item_name}")
                
                gt_base_name = match.group(1) # e.g., real_seq_000_part_1_dif_1
                source_gt_filename = f"{gt_base_name}.txt"
                source_gt_filepath = os.path.join(gt_source_base_path, source_gt_filename)

                if not os.path.isfile(source_gt_filepath):
                    print(f"  Warning: Source GT file '{source_gt_filepath}' not found. Skipping.")
                    continue

                # Create the 'gt' subfolder inside the record folder
                target_gt_subfolder = os.path.join(item_path, "gt")
                try:
                    os.makedirs(target_gt_subfolder, exist_ok=True)
                    print(f"  Ensured directory: {target_gt_subfolder}")
                except OSError as e:
                    print(f"  Error creating directory '{target_gt_subfolder}': {e}. Skipping.")
                    continue
                
                # Define the destination path for the GT file with the new name
                destination_gt_filepath = os.path.join(target_gt_subfolder, "gt_wTc.txt")

                # Copy and rename the GT file
                try:
                    shutil.copy2(source_gt_filepath, destination_gt_filepath)
                    print(f"  Copied '{source_gt_filepath}' to '{destination_gt_filepath}'")
                except Exception as e:
                    print(f"  Error copying file for '{item_name}': {e}")
            # else:
            #     print(f"Folder '{item_name}' does not match pattern. Skipping.") # Optional: for debugging
        # else:
        #     print(f"Item '{item_name}' is not a directory. Skipping.") # Optional: for debugging

    print("\nScript finished.")

if __name__ == "__main__":
    organize_ground_truth_files()