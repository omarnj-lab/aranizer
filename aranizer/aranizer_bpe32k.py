from transformers import PreTrainedTokenizerFast

def get_tokenizer():
    return PreTrainedTokenizerFast(tokenizer_file="C:/Users/Lenovo/Desktop/aranizer/aranizer/BPE_tokenizer/BPE_tokenizer_32.0K.json")
