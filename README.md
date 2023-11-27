# AraNizer

## Description
AraNizer is a sophisticated Python package offering a suite of custom tokenizers specifically designed for Arabic language processing. Built with SentencePiece and Byte Pair Encoding (BPE) methodologies, these tokenizers are engineered to be compatible with the `transformers` and `sentence_transformers` libraries. Each tokenizer in the AraNizer collection is optimized for different NLP tasks, accommodating a diverse range of vocabulary sizes to suit various linguistic scenarios.

## Installation
Install AraNizer effortlessly with pip:
```bash
pip install AraNizer

Usage
Start by importing the desired tokenizer from AraNizer:

from AraNizer import aranizer_bpe32k, aranizer_bpe50k, aranizer_bpe64k, aranizer_bpe86k, aranizer_sp32k, aranizer_sp50k, aranizer_sp64k, aranizer_sp86k

Load your tokenizer:

tokenizer = aranizer_bpe32k.get_tokenizer()  # Replace aranizer_bpe32k with your chosen tokenizer

Example of tokenizing a text:

text = "مثال على النص العربي"  # Example Arabic text
tokens = tokenizer.tokenize(text)
print(tokens)
```

Encoding Text:
To encode text, use the encode method. This converts a text string into a sequence of token ids:
```bash
text = "مثال على النص العربي"  # Example Arabic text
encoded_output = tokenizer.encode(text, add_special_tokens=True)
print(encoded_output)
```
Decoding Text:
To convert token ids back to text, use the decode method:
```bash
decoded_text = tokenizer.decode(encoded_output)
print(decoded_text)
```

AraNizers
```bash
aranizer_bpe32k: Tailored for general language modeling with a 32k vocab size.
aranizer_bpe50k: Ideal for technical or scientific texts, featuring a 50k vocab size.
aranizer_bpe64k: Provides comprehensive language coverage with a 64k vocab size.
aranizer_bpe86k: Suitable for extensive vocabularies in large-scale NLP tasks with an 86k vocab size.
aranizer_sp32k: Efficiently segments Arabic dialects with a 32k vocab size.
aranizer_sp50k: Designed for complex text analysis, equipped with a 50k vocab size.
aranizer_sp64k: Balances performance and breadth in NLP applications with a 64k vocab size.
aranizer_sp86k: Supports multilingual and cross-lingual tasks with an 86k vocab size.
```
Requirements:
- transformers
- sentence_transformers
  
License:
This project is licensed under the MIT License.

Contact:
For queries or assistance, please contact onajar@psu.edu.sa.

Acknowledgments:
Special thanks to Prince Sultan University and Riotu Lab, under the guidance of Dr. Lahouari Ghouti and Dr. Anis Kouba, for their invaluable support.

Version:
aranizer==0.1

Citations:
If AraNizer benefits your research, please cite us:
```bash
@misc{AraNizer,
  title={Aranizer: A Custom Tokenizer for Enhanced Arabic Language Processing},
  author={Omar Najar, Serry Sibaee, Lahouari Ghouti & Anis Kouba. Prince Sultan University, Riyadh, Saudi Arabia},
  year={2023},
  howpublished={\url{https://github.com/omarnj-lab/aranizer}}
}
```


