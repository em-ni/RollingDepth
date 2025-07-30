import os
import subprocess

def process_videos(input_folder, output_folder, target_fps=10):
    """
    Processes all video files in the input_folder, changes their frame rate
    using ffmpeg, and saves them to the output_folder.

    Args:
        input_folder (str): Path to the folder containing input videos.
        output_folder (str): Path to the folder where processed videos will be saved.
        target_fps (int): The target frame rate for the output videos.
    """
    # Ensure output folder exists, create if it doesn't
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # Common video file extensions
    video_extensions = ('.mp4', '.mkv', '.mov', '.avi', '.flv', '.wmv', '.webm')
    processed_count = 0
    skipped_count = 0

    for filename in os.listdir(input_folder):
        input_file_path = os.path.join(input_folder, filename)

        # Check if it's a file and has a recognized video extension
        if os.path.isfile(input_file_path) and filename.lower().endswith(video_extensions):
            base, ext = os.path.splitext(filename)
            output_filename = f"{base}_{target_fps}fps{ext}"
            output_file_path = os.path.join(output_folder, output_filename)

            print(f"Processing: {filename} -> {output_filename}")

            # Construct the ffmpeg command
            # -y: overwrite output files without asking
            command = [
                'ffmpeg',
                '-i', input_file_path,
                '-r', str(target_fps),
                '-y',  # Overwrite output file if it exists
                output_file_path
            ]

            try:
                # Execute the command
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()

                if process.returncode == 0:
                    print(f"Successfully processed: {output_filename}")
                    processed_count += 1
                else:
                    print(f"Error processing {filename}:")
                    print(f"FFmpeg STDOUT: {stdout.decode(errors='ignore')}")
                    print(f"FFmpeg STDERR: {stderr.decode(errors='ignore')}")
                    skipped_count += 1

            except FileNotFoundError:
                print("Error: ffmpeg command not found. Make sure FFmpeg is installed and in your PATH.")
                return
            except Exception as e:
                print(f"An unexpected error occurred with {filename}: {e}")
                skipped_count += 1
        else:
            if os.path.isfile(input_file_path):
                print(f"Skipping non-video file: {filename}")

    print("\n--- Processing Complete ---")
    print(f"Successfully processed: {processed_count} video(s)")
    print(f"Skipped or failed: {skipped_count} file(s)")

if __name__ == "__main__":
    # --- Configuration ---
    INPUT_VIDEO_FOLDER = "sim_original"  # Change this to your input folder path
    OUTPUT_VIDEO_FOLDER = "sim_10fps" # Change this to your desired output folder path
    TARGET_FRAME_RATE = 10
    # ---------------------

    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_input_folder = os.path.join(script_dir, INPUT_VIDEO_FOLDER)
    abs_output_folder = os.path.join(script_dir, OUTPUT_VIDEO_FOLDER)

    if not os.path.isdir(abs_input_folder):
        print(f"Error: Input folder '{abs_input_folder}' not found.")
        print(f"Please create it and put your videos inside, or correct the INPUT_VIDEO_FOLDER path.")
    else:
        process_videos(abs_input_folder, abs_output_folder, TARGET_FRAME_RATE)