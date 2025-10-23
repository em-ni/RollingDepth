import os
import re
import shutil

"""
Script for processing BronchoSim output folder into the required format for RollingDepth training/evaluation.
"""

def process_folder(root_folder):
    print(f"Processing folder: {root_folder}")
    # Loop through all subfolders in the root directory
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Match pattern record_bX_<SOMETHING> where X is a number, and the next part is any string (including a number or float)
        match = re.match(r"record_b(\d+)_.*", folder_name)
        if match:
            # Extract the number X and format it as 00X
            num = int(match.group(1))
            new_name = f"video_{num:03d}"
            new_path = os.path.join(root_folder, new_name)

            # Rename folder
            os.rename(folder_path, new_path)
            print(f"Renamed: {folder_name} -> {new_name}")

            # Process contents of the renamed folder
            for item in os.listdir(new_path):
                item_path = os.path.join(new_path, item)
                # Keep only rgb and depth subfolders
                if item not in ["rgb", "depth"]:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                    print(f"Deleted: {item_path}")

            # Inside 'depth', delete images with "vis" in the filename
            depth_path = os.path.join(new_path, "depth")
            if os.path.exists(depth_path) and os.path.isdir(depth_path):
                for img_name in os.listdir(depth_path):
                    if "vis" in img_name:
                        img_path = os.path.join(depth_path, img_name)
                        os.remove(img_path)
                        print(f"Removed vis file: {img_path}")
        else:
            print(f"Skipping folder: {folder_name} (does not match pattern)")

if __name__ == "__main__":
    folder = "/mnt/c/Users/z5440219/OneDrive - UNSW/Desktop/github/RollingDepth/data/em/sim_branches"
    process_folder(folder)
