from model import GPT
from config import *
from utils import tokenize_text, generate,get_tokenizer,detokenize_text,load_weights


sentence = "Bonjour"
tokenizer = get_tokenizer(VOCAB_PATH)
model = GPT()
model = load_weights(model, "last.pth")

idx = tokenize_text(tokenizer, sentence)
idx = idx

generated = generate(model, idx, 10)

print(detokenize_text(tokenizer, generated))
