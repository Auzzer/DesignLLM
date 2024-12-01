from typing import Dict, Iterator, List

from tiktoken import Encoding
from tiktoken.load import load_tiktoken_bpe
from torchtune.modules.tokenizers._utils import BaseTokenizer
import torchtune
tokenizer = torchtune.modules.tokenizers._tiktoken.TikTokenBaseTokenizer("/projectnb/ece601/24FallA2Group16/checkpoints/Llama3.1-8B-Instruct/tokenizer.model")
tokenized_text = tokenizer.encode("Hello world!", add_bos=True, add_eos=True)