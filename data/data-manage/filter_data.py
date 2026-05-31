# filter_data.py
INPUT  = "data/dataset/300Ktrain3_shuffled.txt"
OUTPUT = "data/dataset/300Ktrain3_clean.txt"

MIN_SCORE  = 0.70   # Increased from 0.65 → filter out MT noise
MAX_EN_WORDS = 100  # Prevent mid-sentence truncation
MAX_JA_CHARS = 120

kept = 0
with open(INPUT, 'r', encoding='utf-8') as fin, \
     open(OUTPUT, 'w', encoding='utf-8') as fout:
    for line in fin:
        parts = line.strip('\n').split('\t')
        if len(parts) < 5:
            continue
        try:
            score = float(parts[2])
        except:
            continue
        en, ja = parts[3], parts[4]
        if score < MIN_SCORE:
            continue
        if len(en.split()) > MAX_EN_WORDS:
            continue
        if len(ja) > MAX_JA_CHARS:
            continue
        fout.write(line)
        kept += 1

print(f"Kept: {kept:,} / 300,000 lines")