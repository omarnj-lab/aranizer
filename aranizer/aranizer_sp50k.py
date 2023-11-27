from transformers import PreTrainedTokenizerFast

def get_tokenizer():
    # Initialize the tokenizer
    tokenizer_fast = PreTrainedTokenizerFast(tokenizer_file="sentence peice tokenizers/SP_tokenizer_50.0K.json")

    # List of Arabic diacritics
    arabic_diacritics = ['َ', 'ً', 'ُ', 'ِ', 'ٍ', 'ْ', 'ّ', 'ٓ', '٭', 'ء']

    # Add Arabic diacritics to the tokenizer's vocabulary as special tokens
    num_added_toks = tokenizer_fast.add_tokens(arabic_diacritics, special_tokens=True)

    return tokenizer_fast
