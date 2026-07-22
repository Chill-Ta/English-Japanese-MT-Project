import os
import random
import time

# ==========================================
# 1. Configure File Paths and Target Line Range
# ==========================================
INPUT_FILE = "data/dataset/train-set/en-ja/resources/en-ja_extracted_filtered.txt" 
OUTPUT_FILE = "data/dataset/train-set/en-ja/resources/en-ja_extracted4M.txt"

# Set target rows to extract 
START_ROW = 1      # Starting line number
END_ROW = 4000000        # Ending line number (set to None to extract until the end of the file)

def extract_and_shuffle():
    if END_ROW is not None:
        print(f"Extracting data from line {START_ROW:,} to {END_ROW:,} from file '{INPUT_FILE}'...")
    else:
        print(f"Extracting data from line {START_ROW:,} to the end of the file...")
        
    start_time = time.time()
    extracted_lines = []
    
    try:
        # 1. Read and extract only the target lines
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                current_row = i + 1 # Start counting lines from 1
                
                # If within the target range, append to list
                if current_row >= START_ROW:
                    extracted_lines.append(line)
                
                # If exceeding the ending row, stop reading the file immediately 
                if END_ROW is not None and current_row >= END_ROW:
                    break
                    
        total_extracted = len(extracted_lines)
        
        if total_extracted == 0:
            print("No data found within the specified line range. Please check the line count of the source file.")
            return
            
        print(f"Successfully extracted {total_extracted:,} lines.")
        
        # 2. Shuffle data
        print("Shuffling data...")
        random.shuffle(extracted_lines)
        
        # 3. Save to a new file
        print("Saving file...")
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.writelines(extracted_lines)
            
        print(f"Completed! The file is ready at: {OUTPUT_FILE}")
        print(f"Total time elapsed: {time.time() - start_time:.2f} seconds")
        
    except FileNotFoundError:
        print(f"Error: Source file '{INPUT_FILE}' not found. Please double-check the file path!")

if __name__ == "__main__":
    extract_and_shuffle()