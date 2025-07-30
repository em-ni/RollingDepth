import os
import numpy as np
from PIL import Image, UnidentifiedImageError
import argparse

def read_depth_png(png_path, ref_shape=None):
    try:
        img = Image.open(png_path)
        arr = np.array(img)
        # If image has 3 channels, convert to grayscale (average or use PIL)
        if arr.ndim == 3 and arr.shape[2] == 3:
            # Use PIL to convert to grayscale for consistency
            img_gray = img.convert('L')
            arr = np.array(img_gray)
        arr = arr.astype(np.float32)
        if ref_shape is not None and arr.shape != ref_shape:
            print(f"Warning: {png_path} has shape {arr.shape}, expected {ref_shape}. Skipping.")
            return None
        return arr
    except (OSError, UnidentifiedImageError) as e:
        print(f"Warning: Could not read {png_path}: {e}")
        return None

def pngs_to_npy_memmap(input_folder, output_path, sort_key=None):
    png_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.png')]
    if not png_files:
        raise ValueError(f"No .png files found in {input_folder}")
    if sort_key is not None:
        png_files.sort(key=sort_key)
    else:
        png_files.sort()

    # First pass: determine shape and count valid images
    ref_shape = None
    valid_files = []
    for fname in png_files:
        fpath = os.path.join(input_folder, fname)
        if os.path.getsize(fpath) == 0:
            print(f"Warning: {fpath} is empty, skipping.")
            continue
        arr = read_depth_png(fpath)
        if arr is not None:
            if ref_shape is None:
                ref_shape = arr.shape
            if arr.shape == ref_shape:
                valid_files.append(fname)
    n_valid = len(valid_files)
    if n_valid == 0:
        raise RuntimeError("No valid PNG files could be loaded.")

    # Create memmap array
    mmap_arr = np.lib.format.open_memmap(
        output_path, mode='w+', dtype=np.float32, shape=(n_valid, *ref_shape)
    )

    # Second pass: write images to memmap
    idx = 0
    for fname in valid_files:
        fpath = os.path.join(input_folder, fname)
        arr = read_depth_png(fpath, ref_shape=ref_shape)
        if arr is not None:
            mmap_arr[idx] = arr
            idx += 1
    print(f"Saved {n_valid} frames to {output_path} with shape {mmap_arr.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a folder of depth .png files to a .npy file (memory efficient).")
    parser.add_argument("input_folder", type=str, help="Folder containing .png depth images")
    parser.add_argument("output_npy", type=str, help="Output .npy file path")
    args = parser.parse_args()
    pngs_to_npy_memmap(args.input_folder, args.output_npy)
