# Custom Tokenizers

## Description
AraNizer is a Python package providing a collection of tokenizers built with SentencePiece and BPE for Arabic Language, compatible with the `transformers` and `sentence_transformers` libraries. These tokenizers are designed for various NLP tasks and have different vocabulary sizes.

## Installation
To install Custom Tokenizers, simply run:

```bash
pip install AraNizer

Usage
To use a tokenizer in your project, first import the package:

from AraNizer import aranizer_bpe32k, aranizer_bpe64k, aranizer_bpe50k, aranizer_bpe86k , aranizer_sp32k,aranizer_sp64k, aranizer_sp50k, aranizer_sp86k

Then, you can load your desired tokenizer:

aranizer_bpe32k = aranizer_bpe32k.get_tokenizer()
aranizer_bpe50k = aranizer_bpe50k.get_tokenizer()
aranizer_bpe64k = aranizer_bpe64k.get_tokenizer()
aranizer_bpe86k = aranizer_bpe86k.get_tokenizer()

aranizer_sp32k = aranizer_sp32k.get_tokenizer()
aranizer_sp50k = aranizer_sp50k.get_tokenizer()
aranizer_sp64k = aranizer_sp64k.get_tokenizer()
aranizer_sp86k = aranizer_sp86k.get_tokenizer()


AraNizers:

aranizer_bpe32k: Custom BPE Tokenizer with a vocabulary size of 32k for Arabic language, optimized for general language modeling.

aranizer_bpe50k: Custom BPE Tokenizer with a vocabulary size of 50k, ideal for specialized domains like technical or scientific texts.

aranizer_bpe64k: Custom BPE Tokenizer with a vocabulary size of 64k, designed for comprehensive language coverage, including rare words and phrases.

aranizer_bpe86k: Custom BPE Tokenizer with a vocabulary size of 86k, suitable for large-scale NLP tasks requiring extensive vocabulary.

aranizer_sp32k: Custom SentencePiece Tokenizer with a vocabulary size of 32k, offering efficient segmentation for Arabic dialects.

aranizer_sp50k: Custom SentencePiece Tokenizer with a vocabulary size of 50k, tailored for complex text analysis and interpretation.

aranizer_sp64k: Custom SentencePiece Tokenizer with a vocabulary size of 64k, balancing performance and comprehensiveness for diverse NLP applications.

aranizer_sp86k: Custom SentencePiece Tokenizer with a vocabulary size of 86k, providing robust support for multilingual and cross-lingual tasks.

Requirements:

transformers
sentence_transformers


License:
This project is licensed under the MIT License.

Contact:
For any queries or assistance, please reach out to onajar@psu.edu.sa.

Acknowledgments
Thanks to Prince Sultan University, especially, Riotu Lab for their support under supervision of Dr. Lahouari Ghouti and Dr. Anis Kouba.

Version:

aranizer==0.1

Citations:
If you use AraNizer in your research, please cite:

@misc{AraNizer,
  title={Aranizer:A Custom Tokenizer for Enhanced Arabic Language Processing},
  author={Omar Najar, Serry Sibaee, Lahouari Ghouti & Dr. Anis Kouba. Prince Sultan University, Riyadh, Saudi Arabia},
  year={2023},
  howpublished={\url{https://github.com/yourusername/custom_tokenizers}}
}

