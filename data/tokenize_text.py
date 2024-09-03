from tokenizers import Tokenizer
import numpy as np
import os
import time

from config import ENCODING, TOKENS_DIR, VOCAB_PATH
from utils import get_tokenizer

data_path = "/media/SSD2/LLM/data/src/fr_clean.txt"
# data_path = "subset.txt"
token_per_file = 150_000_000
test_split = 0.2


def split_encoded_text(src_path: str, tokenizer: Tokenizer, tok_per_file: int, test_ratio: float) -> None:
    tokens_accumulated = []
    file_index = 0
    tokens_for_test = []

    # with open(src_path, "r", encoding=ENCODING) as src_file:
    #    total_lines = src_file.readlines()
    #   total_tokens = 0
    #    for line in total_lines:
    #        total_tokens += len(tokenizer.encode(line.strip()).ids)
    #    train_limit = int((1 - test_ratio) * total_tokens)
    with open(src_path, "r", encoding=ENCODING) as src_file:
        current_token_count = 0
        for line in src_file:
            encoded = tokenizer.encode(line.strip())
            tokens_accumulated.extend(encoded.ids)
            current_token_count += len(encoded.ids)
            if len(tokens_accumulated) >= tok_per_file:
                tokens_to_save = tokens_accumulated[:tok_per_file]
                save_npy_file(os.path.join(TOKENS_DIR, f"train_tokens_{file_index}.npy"), np.array(tokens_to_save))
                tokens_accumulated = tokens_accumulated[tok_per_file:]
                file_index += 1
        if tokens_accumulated:
            save_npy_file(os.path.join(TOKENS_DIR, f"train_tokens_{file_index}.npy"), np.array(tokens_accumulated))
    """
    file_index = 0
    while len(tokens_for_test) > 0:
        if len(tokens_for_test) > tok_per_file:
            save_npy_file(os.path.join(TOKENS_DIR, f"test_tokens_{file_index}.npy"),
                          np.array(tokens_for_test[:tok_per_file]))
            tokens_for_test = tokens_for_test[tok_per_file:]
        else:
            save_npy_file(os.path.join(TOKENS_DIR, f"test_tokens_{file_index}.npy"), np.array(tokens_for_test))
            tokens_for_test = []
        file_index += 1
"""


def save_npy_file(path: str, data) -> None:
    np.save(path, data)
    print(f"saved:{path}")


if __name__ == "__main__":
    b = time.time()
    print("launch tokenizations.....")
    tok = get_tokenizer(VOCAB_PATH)
    split_encoded_text(data_path, tok, token_per_file, test_split)
    e = time.time()
    print("ended.")
    print(f"time:{e - b} s")
