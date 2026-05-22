from tokenizers import Tokenizer

def verify_tokenizer(file_path):
    print(f"Loading Tokenizer from: {file_path}")
    tokenizer = Tokenizer.from_file(file_path)
    
    # ==========================================
    # 1. Check Vocabulary Size (Vocab Size)
    # ==========================================
    vocab_size = tokenizer.get_vocab_size()
    print(f"\n1. Vocab size: {vocab_size:,} words")
    if vocab_size < 1000:
        print("Warning: Vocab size is too small. Please check your corpus again")
        
    # ==========================================
    # 2. Check Special Tokens (for bidirectional translation)
    # ==========================================
    print("\n2. Special Tokens Check:")
    special_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<mask>", "<2en>", "<2ja>"]
    all_special_found = True
    
    for token in special_tokens:
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            print(f"Not found: {token}")
            all_special_found = False
        else:
            print(f"{token:<6} -> ID: {token_id}")
            
    if not all_special_found:
        print("Warning: Special Tokens are not completed. Model might be crashed")

    # ==========================================
    # 3. Tokenization Test (Metaspace & BPE)
    # ==========================================
    print("\n3. Tokenization Test (Metaspace & BPE)")
    
    # 3.1 English (EN)
    en_text = "Hello, how are you?"
    en_encoded = tokenizer.encode(en_text)
    print(f"\n🇺🇸 Original Text EN: {en_text}")
    print(f"   Tokens : {en_encoded.tokens}")
    
    # 3.2 Japanese (JA)
    ja_text = "こんにちは、世界！それはテストです。"
    ja_encoded = tokenizer.encode(ja_text)
    print(f"\n🇯🇵 Original Text JA: {ja_text}")
    print(f"   Tokens : {ja_encoded.tokens}")

if __name__ == "__main__":
    verify_tokenizer("data/shared_tokenizer.json")