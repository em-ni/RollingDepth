import os
import subprocess
import sys
import argparse
import shutil

def convert_to_seconds(time_str):
    """Converts M:S or MM:SS format to total seconds."""
    try:
        parts = time_str.strip().split(':')
        if len(parts) != 2:
            raise ValueError("Timestamp must be in M:S format.")
        minutes = int(parts[0])
        seconds = int(parts[1]) # Or float(parts[1]) if you need fractions
        if seconds < 0 or seconds >= 60:
             raise ValueError("Seconds must be between 0 and 59.")
        if minutes < 0:
             raise ValueError("Minutes cannot be negative.")
        total_seconds = (minutes * 60) + seconds
        return total_seconds
    except ValueError as e:
        print(f"Error converting timestamp '{time_str}': {e}")
        sys.exit(1) # Exit if timestamp is invalid

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
            sys.exit(1)
        duration = float(duration_str)
        if duration <= 0:
            print(f"Error: Video duration reported by ffprobe is not positive ({duration:.2f}s) for '{video_path}'.")
            sys.exit(1)
        return duration
    except subprocess.CalledProcessError as e:
        print(f"Error getting video duration using ffprobe for '{video_path}':")
        print(f"FFprobe STDERR:\n{e.stderr.strip()}")
        sys.exit(1)
    except ValueError:
        # result might not be defined if subprocess.run failed before assignment
        output_for_error = "N/A"
        if 'result' in locals() and hasattr(result, 'stdout'):
            output_for_error = result.stdout.strip()
        print(f"Error: Could not parse duration from ffprobe output ('{output_for_error}') for '{video_path}'.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while getting video duration for '{video_path}': {e}")
        sys.exit(1)

def split_video(video_path, timestamps, output_dir, output_prefix="chunk"):
    """
    Splits the video into chunks.
    User-provided timestamps T1, T2, ... Tn define cut points.
    Segments created: 0-T1, T1-T2, ..., Tn-1-Tn, Tn-end_of_video.
    A single timestamp T results in segments 0-T and T-end_of_video.
    """
    check_ffmpeg()
    check_ffprobe() # Added ffprobe check

    if not os.path.isfile(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    try:
        user_cut_points_sec = sorted([convert_to_seconds(ts) for ts in timestamps])
    except Exception:
         sys.exit(1) # Error already printed in convert_to_seconds

    if not user_cut_points_sec:
        print("Error: No valid timestamps provided for cutting.")
        sys.exit(1)

    video_duration_sec = get_video_duration(video_path)
    print(f"Video duration: {video_duration_sec:.2f} seconds.")

    # Determine the actual end points for each segment
    segment_end_points = []
    for point in user_cut_points_sec:
        if 0 < point < video_duration_sec:
            segment_end_points.append(point)
        elif point >= video_duration_sec:
            print(f"Info: Timestamp {point}s is at or beyond video duration ({video_duration_sec:.2f}s). Effective cut will be at video end.")
        elif point <= 0:
            print(f"Warning: Timestamp {point}s is zero or negative, ignoring.")

    segment_end_points = sorted(list(set(segment_end_points))) # Unique, sorted, valid cut points

    # Ensure video_duration_sec is the final end point if not already covered
    if not segment_end_points or segment_end_points[-1] < video_duration_sec:
        segment_end_points.append(video_duration_sec)
    
    # Clean up: ensure all points are positive and unique after potentially adding video_duration_sec
    segment_end_points = sorted(list(set(p for p in segment_end_points if p > 0)))

    if not segment_end_points:
        print(f"Error: No valid segment end points could be determined. Video duration: {video_duration_sec:.2f}s.")
        sys.exit(1)
    
    start_times_list = [0] + segment_end_points[:-1]
    end_times_list = segment_end_points

    base_name, extension = os.path.splitext(os.path.basename(video_path))
    if not extension:
        extension = ".mp4"
        print(f"Warning: Input video has no extension, assuming '{extension}' for output.")

    print(f"Starting video splitting for '{video_path}'...")
    # print(f"Target segment end points: {segment_end_points}") # For debugging
    # print(f"Calculated start times: {start_times_list}")      # For debugging
    # print(f"Calculated end times: {end_times_list}")        # For debugging

    created_files_count = 0
    for i, (start_sec, end_sec) in enumerate(zip(start_times_list, end_times_list)):
        if start_sec >= end_sec:
            print(f"Warning: Skipping segment {i+1} because start time ({start_sec:.2f}s) is not before end time ({end_sec:.2f}s).")
            continue

        output_filename = f"{output_prefix}_{i+1:03d}{extension}"
        output_path = os.path.join(output_dir, output_filename)

        print(f"Creating chunk {i+1}: Start={start_sec:.2f}s, End={end_sec:.2f}s -> '{output_filename}'")

        command = [
            'ffmpeg',
            '-i', video_path,
            '-ss', str(start_sec),
            '-to', str(end_sec),
            '-c', 'copy',
            '-avoid_negative_ts', 'make_zero',
            '-y',
            output_path
        ]

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Successfully created '{output_path}'")
            created_files_count +=1
        except subprocess.CalledProcessError as e:
            print(f"\nError running FFmpeg for chunk {i+1}:")
            print(f"Command: {' '.join(e.cmd)}")
            print(f"Return Code: {e.returncode}")
            print(f"FFmpeg STDERR:\n{e.stderr}")
            print("\nStopping script.")
            sys.exit(1)
        except Exception as e:
            print(f"\nAn unexpected error occurred during FFmpeg execution for chunk {i+1}: {e}")
            print("\nStopping script.")
            sys.exit(1)

    if created_files_count == 0:
        print("\nNo video segments were created. Please check your timestamps and video duration.")
    else:
        print("\nVideo splitting complete.")
        print(f"Chunks saved in: '{os.path.abspath(output_dir)}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a video into chunks based on timestamps (M:S format).")
    parser.add_argument("video_file", help="Path to the input video file.")
    parser.add_argument("timestamps", nargs='+', help="List of end timestamps for chunks (e.g., '1:30' '3:45' '5:10').")
    parser.add_argument("-o", "--output-dir", default="video_chunks", help="Directory to save the output chunks (default: ./video_chunks).")
    parser.add_argument("-p", "--prefix", default="chunk", help="Prefix for output chunk filenames (default: chunk).")

    args = parser.parse_args()

    split_video(args.video_file, args.timestamps, args.output_dir, args.prefix)