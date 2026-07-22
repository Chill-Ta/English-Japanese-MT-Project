import os

def split_data_into_chunks(input_filepath, output_directory, lines_per_chunk):
    """
    Splits a massive text file into smaller chunk files line by line.
    This streaming approach prevents Out-Of-Memory (OOM) errors, making it 
    capable of handling datasets of any size.
    """
    # Verify input file exists
    if not os.path.exists(input_filepath):
        print(f"Error: Input file '{input_filepath}' not found.")
        return

    # Create output directory if it does not exist
    os.makedirs(output_directory, exist_ok=True)

    chunk_index = 1
    current_line_count = 0
    current_chunk_file = None

    print(f"Starting to split: {input_filepath}")
    print(f"Target lines per chunk: {lines_per_chunk:,}")
    print("-" * 50)

    try:
        # Open the large input file for reading
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            for line in infile:
                # Open a new chunk file when starting or reaching the limit
                if current_line_count % lines_per_chunk == 0:
                    
                    # Close the previous file before opening a new one
                    if current_chunk_file:
                        current_chunk_file.close()

                    # Format chunk filename (e.g., chunk_01.txt, chunk_02.txt)
                    chunk_filename = f"chunk_{chunk_index:02d}.txt"
                    chunk_filepath = os.path.join(output_directory, chunk_filename)
                    
                    current_chunk_file = open(chunk_filepath, 'w', encoding='utf-8')
                    print(f"Writing to {chunk_filename}...")
                    
                    chunk_index += 1

                # Write the current line to the active chunk file
                current_chunk_file.write(line)
                current_line_count += 1

    except Exception as e:
        print(f"An error occurred during splitting: {e}")
        
    finally:
        # Ensure the last file is properly closed when loop finishes or breaks
        if current_chunk_file:
            current_chunk_file.close()

    total_chunks = chunk_index - 1
    print("-" * 50)
    print("Process Completed Successfully.")
    print(f"Total lines processed: {current_line_count:,}")
    print(f"Total chunks created:  {total_chunks}")


if __name__ == "__main__":
    # ==========================================
    # CLI Configuration (Phase 3 upgrade)
    # ==========================================
    import argparse
    parser = argparse.ArgumentParser(
        description="Split a large dataset file into smaller chunks for training."
    )
    parser.add_argument("--input", type=str,
                        default="data/dataset/train-set/en-ja/resources/en-ja_extracted4M.txt",
                        help="Input file path")
    parser.add_argument("--output_dir", type=str,
                        default="data/dataset/chunks",
                        help="Output directory for chunk files")
    parser.add_argument("--lines_per_chunk", type=int, default=250000,
                        help="Number of lines per chunk (default: 250000, use 500000 for 5M+ datasets)")
    args = parser.parse_args()

    split_data_into_chunks(args.input, args.output_dir, args.lines_per_chunk)