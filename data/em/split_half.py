import os
import subprocess
import sys
import argparse
import shutil

def check_ffmpeg():
    """Checks if ffmpeg command exists."""
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg command not found.")
        print("Please install FFmpeg and ensure it's in your system's PATH.")
        sys.exit(1)

def check_ffprobe():
    """Checks if ffprobe command exists."""
    if shutil.which("ffprobe") is None:
        print("ERROR: ffprobe command not found. It is usually part of FFmpeg installation.")
        print("Please ensure FFmpeg (and ffprobe) is installed and in your system's PATH.")
        sys.exit(1)

def get_video_duration(video_path):
    """Gets the duration of the video in seconds using ffprobe."""
    command = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        duration_str = result.stdout.strip()
        if not duration_str:
            print(f"Error: ffprobe returned empty duration for '{video_path}'.")
            return None # Changed to return None on error
        duration = float(duration_str)
        if duration <= 0:
            print(f"Error: Video duration reported by ffprobe is not positive ({duration:.2f}s) for '{video_path}'.")
            return None # Changed to return None on error
        return duration
    except subprocess.CalledProcessError as e:
        print(f"Error getting video duration using ffprobe for '{video_path}':")
        print(f"FFprobe STDERR:\n{e.stderr.strip()}")
        return None # Changed to return None on error
    except ValueError:
        output_for_error = "N/A"
        if 'result' in locals() and hasattr(result, 'stdout'):
            output_for_error = result.stdout.strip()
        print(f"Error: Could not parse duration from ffprobe output ('{output_for_error}') for '{video_path}'.")
        return None # Changed to return None on error
    except Exception as e:
        print(f"An unexpected error occurred while getting video duration for '{video_path}': {e}")
        return None # Changed to return None on error

def split_video_in_half(video_path, output_dir):
    """
    Splits a single video into two halves.
    Part 1: 0 to midpoint
    Part 2: midpoint to end_of_video
    """
    base_name, extension = os.path.splitext(os.path.basename(video_path))
    if not extension:
        extension = ".mp4" # Default extension if none found
        print(f"Warning: Input video '{video_path}' has no extension, assuming '{extension}' for output.")

    video_duration_sec = get_video_duration(video_path)
    if video_duration_sec is None:
        print(f"Skipping '{video_path}' due to issues in getting duration.")
        return False

    if video_duration_sec <= 0:
        print(f"Skipping '{video_path}' as its duration is not positive ({video_duration_sec:.2f}s).")
        return False

    midpoint_sec = video_duration_sec / 2.0
    print(f"Processing '{video_path}': Duration={video_duration_sec:.2f}s, Midpoint={midpoint_sec:.2f}s")

    output_path_part1 = os.path.join(output_dir, f"{base_name}_part1{extension}")
    output_path_part2 = os.path.join(output_dir, f"{base_name}_part2{extension}")

    # Command for the first half
    command1 = [
        'ffmpeg',
        '-i', video_path,
        '-ss', '0',
        '-to', str(midpoint_sec),
        '-c', 'copy',
        '-avoid_negative_ts', 'make_zero',
        '-y', # Overwrite output files without asking
        output_path_part1
    ]

    # Command for the second half
    command2 = [
        'ffmpeg',
        '-i', video_path,
        '-ss', str(midpoint_sec),
        '-to', str(video_duration_sec),
        '-c', 'copy',
        '-avoid_negative_ts', 'make_zero',
        '-y', # Overwrite output files without asking
        output_path_part2
    ]

    success = True
    for i, (cmd, out_path) in enumerate(zip([command1, command2], [output_path_part1, output_path_part2])):
        part_num = i + 1
        print(f"Creating part {part_num} for '{base_name}{extension}' -> '{os.path.basename(out_path)}'")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"Successfully created '{out_path}'")
        except subprocess.CalledProcessError as e:
            print(f"\nError running FFmpeg for part {part_num} of '{video_path}':")
            print(f"Command: {' '.join(e.cmd)}")
            print(f"Return Code: {e.returncode}")
            print(f"FFmpeg STDERR:\n{e.stderr.strip()}")
            success = False
            break # Stop processing this video if a part fails
        except Exception as e:
            print(f"\nAn unexpected error occurred during FFmpeg execution for part {part_num} of '{video_path}': {e}")
            success = False
            break # Stop processing this video if a part fails
    return success

def process_folder(input_folder, output_folder):
    """
    Processes all videos in the input_folder and saves split halves to output_folder.
    """
    check_ffmpeg()
    check_ffprobe()

    if not os.path.isdir(input_folder):
        print(f"Error: Input folder not found at '{input_folder}'")
        sys.exit(1)

    os.makedirs(output_folder, exist_ok=True)

    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv') # Add more if needed
    videos_processed_count = 0
    videos_skipped_count = 0

    print(f"Scanning for videos in '{input_folder}'...")
    for item in os.listdir(input_folder):
        item_path = os.path.join(input_folder, item)
        if os.path.isfile(item_path) and item.lower().endswith(video_extensions):
            print(f"\nFound video: '{item}'")
            if split_video_in_half(item_path, output_folder):
                videos_processed_count += 1
            else:
                videos_skipped_count +=1
        elif os.path.isfile(item_path):
            print(f"Skipping non-video file: '{item}'")


    print("\n--- Processing Summary ---")
    if videos_processed_count > 0:
        print(f"Successfully processed and split {videos_processed_count} video(s).")
        print(f"Split videos saved in: '{os.path.abspath(output_folder)}'")
    else:
        print("No videos were successfully processed and split.")
    if videos_skipped_count > 0:
        print(f"{videos_skipped_count} video(s) were skipped due to errors or invalid format.")
    if videos_processed_count == 0 and videos_skipped_count == 0:
        print(f"No video files found in '{input_folder}' with extensions: {', '.join(video_extensions)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split all videos in a folder into two halves.")
    parser.add_argument("input_folder", help="Path to the folder containing input video files.")
    parser.add_argument("-o", "--output-folder", default="video_halves",
                        help="Directory to save the output video halves (default: ./video_halves).")

    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder)