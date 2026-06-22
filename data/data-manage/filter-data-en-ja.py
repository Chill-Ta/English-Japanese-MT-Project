import re

INPUT  = "data/dataset/train-set/en-ja/resources/en-ja.bicleaner05.txt"
OUTPUT = "data/dataset/train-set/en-ja/resources/en-ja_extracted_filtered.txt"

MIN_SCORE  = 0.70   
MAX_EN_WORDS = 100  
MAX_JA_CHARS = 120
MAX_LENGTH_RATIO = 3.0  # Prevents severe translation omissions or extreme hallucinations

def clean_text(text):
    """
    Apply regex formatting to remove HTML tags, normalize whitespaces, 
    and clean up punctuation artifacts.
    """
    text = re.sub(r'<[^>]+>', '', text)          # Remove HTML tags
    text = re.sub(r'\s+', ' ', text).strip()     # Normalize multiple spaces to single space
    return text

kept = 0
seen_sentences = set()

print("Starting rigorous data cleaning process...")

with open(INPUT, 'r', encoding='utf-8') as fin, \
     open(OUTPUT, 'w', encoding='utf-8') as fout:
    for line in fin:
        parts = line.strip('\n').split('\t')
        
        # Rule 1: Structure verification
        if len(parts) < 5:
            continue
            
        try:
            score = float(parts[2])
        except ValueError:
            continue
            
        # Rule 2: Minimum quality score
        if score < MIN_SCORE:
            continue
            
        # Rule 3: Text cleaning and formatting
        en = clean_text(parts[3])
        ja = clean_text(parts[4])
        
        en_words = len(en.split())
        ja_chars = len(ja)
        
        # Rule 4: Empty string or length limits check
        if en_words == 0 or ja_chars == 0:
            continue
        if en_words > MAX_EN_WORDS or ja_chars > MAX_JA_CHARS:
            continue
            
        # Rule 5: Length Ratio Filtering
        # Approximating that 1 English word usually translates to 2-3 Japanese characters
        if (ja_chars / en_words) > MAX_LENGTH_RATIO * 3 or (en_words / ja_chars) > MAX_LENGTH_RATIO:
            continue
            
        # Rule 6: Exact Deduplication (Removes repetitive loops)
        unique_key = f"{en}|||{ja}"
        if unique_key in seen_sentences:
            continue
        seen_sentences.add(unique_key)
        
        # Write cleaned and verified data to output
        parts[3] = en
        parts[4] = ja
        fout.write('\t'.join(parts) + '\n')
        kept += 1

print(f"Filtering completed. Kept: {kept:,} strictly cleaned lines.")