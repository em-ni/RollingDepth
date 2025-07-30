import sys

def concatenate_files(output_path, input_paths):
    """
    Concatenates multiple input files into a single output file.
    
    Args:
        output_path (str): The path to the file where the result will be saved.
        input_paths (list): A list of strings, where each is a path to an input file.
    """
    print(f"Creating output file: {output_path}")
    
    try:
        # Step 1: Open the output file in 'write' mode ('w').
        # This will create the file or overwrite it if it already exists.
        with open(output_path, 'w', encoding='utf-8') as f_out:
            
            # Step 2: Loop through each of the input file paths.
            for input_path in input_paths:
                try:
                    # Step 3: Open the current input file in 'read' mode ('r').
                    print(f"  -> Appending content from: {input_path}")
                    with open(input_path, 'r', encoding='utf-8') as f_in:
                        # Step 4: Read the content and write it to the output file.
                        # This is a memory-efficient way to copy file contents.
                        for line in f_in:
                            f_out.write(line)
                            
                    # Note: If you want a blank line separating the content of each file,
                    # you could add this line here:
                    # f_out.write('\n')

                except FileNotFoundError:
                    # If an input file is not found, print a warning and stop.
                    print(f"\nError: Input file not found: '{input_path}'")
                    sys.exit(1)

        print("\nSuccess! Files have been joined.")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # The script needs at least 3 arguments:
    # script_name.py, output_file.txt, input_file1.txt
    if len(sys.argv) < 3:
        print("\nUsage: python join_files.py <output_file.txt> <input1.txt> <input2.txt> ...")
        print("Example: python join_files.py merged.txt fileA.txt fileB.txt\n")
        sys.exit(1)

    # The first argument is the output file
    output_file = sys.argv[1]
    
    # All subsequent arguments are the input files
    input_files = sys.argv[2:]

    concatenate_files(output_file, input_files)