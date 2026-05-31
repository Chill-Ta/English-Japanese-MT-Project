import os

# ==========================================
# Configure Source and Target Files
# ==========================================
EN_FILE = "data/dataset/test.en"   # Point to English file
JA_FILE = "data/dataset/test.ja"   # Point to Japanese file
OUTPUT_FILE = "data/test_merged.txt" # Output file to be used

def merge_files():
    print("Reading and pairing files...")
    
    try:
        with open(EN_FILE, 'r', encoding='utf-8') as f_en, \
             open(JA_FILE, 'r', encoding='utf-8') as f_ja, \
             open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
             
            count = 0
            # Use zip() to pull lines from both files concurrently
            for en_line, ja_line in zip(f_en, f_ja):
                
                # 1. Process English (strip leading/trailing whitespaces only)
                en_clean = en_line.strip()
                
                # 2. Process Japanese (remove all spaces in the middle to make it a normal sentence)
                ja_clean = ja_line.replace(' ', '').strip()
                
                # 3. Combine and separate with a Tab (\t)
                f_out.write(f"{en_clean}\t{ja_clean}\n")
                count += 1
                
        print(f"Successfully merged {count:,} sentences!")
        print(f"File ready for evaluate.py at: {OUTPUT_FILE}")
        
    except FileNotFoundError as e:
        print(f"Error: File not found. Please check your paths again ({e})")

if __name__ == "__main__":
    merge_files()