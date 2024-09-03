import os
import time

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from config import *

# Path to the original large data file
data_file = "/media/SSD2/LLM/data/src/fr_clean.txt"
# Path to the temporary smaller data file
subset_file = "subset.txt"
# Output path for the trained tokenizer
out_path = "vocab.json"

nb_line = 10_000

# Create a smaller subset of the data
with open(data_file, 'r', encoding='utf-8') as infile, open(subset_file, 'w+', encoding='utf-8') as outfile:
    for i, line in enumerate(infile):
        if i >= nb_line:  # Use only the first 100,000 lines
            break
        outfile.write(line)
print("Created a subset of the data.")

# Initialize the tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# Trainer configuration
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=VOCAB_SIZE, show_progress=True)

if __name__ == "__main__":
    start = time.time()
    # Train the tokenizer on the smaller subset
    tokenizer.train([subset_file], trainer)
    # Save the trained tokenizer
    tokenizer.save(out_path)
    # Delete the temporary file
    os.remove(subset_file)
    end = time.time()
    print(f"Training took {end - start:.2f} seconds.")
    print("Trained tokenizer and saved to vocab.json.")