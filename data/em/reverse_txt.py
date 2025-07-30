import sys

def reverse_file(input_path, output_path):
    """
    Reads an input file, reverses the order of its lines,
    and writes the result to an output file.
    """
    print(f"Reading from: {input_path}")
    print(f"Writing to:   {output_path}")

    try:
        # Step 1: Read all lines from the input file into a list
        with open(input_path, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()

        # Step 2: Reverse the list of lines
        # The [::-1] slice is a quick and Pythonic way to reverse a list
        reversed_lines = lines[::-1]

        # Step 3: Write the reversed lines to the output file
        with open(output_path, 'w', encoding='utf-8') as f_out:
            f_out.writelines(reversed_lines)

        print("\nSuccess! File lines have been reversed.")

    except FileNotFoundError:
        print(f"\nError: The input file '{input_path}' was not found.")
        sys.exit(1) # Exit with an error code
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Check if the user provided the two required arguments
    if len(sys.argv) != 3:
        print("Usage: python reverse_lines.py <input_file.txt> <output_file.txt>")
        sys.exit(1) # Exit with an error code

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    reverse_file(input_file, output_file)