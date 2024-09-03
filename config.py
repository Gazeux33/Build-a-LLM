import torch
import sys

print(f"python_version: {sys.version}")
print(f"torch_version: {torch.__version__}")
print(f"device_name: {torch.cuda.get_device_name()}")

# *** PATHS ***
DATA_DIR = "data"
VOCAB_PATH = "/media/SSD2/LLM/data/vocab.json"
ENCODING = "utf-8"
TOKENS_DIR = "/media/SSD2/LLM/data/tokens"
MODEL_DIR = "models"

# *** DEVICE ***
device_name = "cpu"
if torch.cuda.is_available():
    device_name = "cuda"
DEVICE = torch.device(device_name)
print(f"using device: {DEVICE}")

# *** TRAIN ***
block_size = 1024  # 1024
batch_size = 4  # 32
total_batch_size = 131072  # in tokens
betas = (0.9, 0.95)
eps = 1e-8
weight_decay = 0.1
EPOCHS = 2
EVAL_FREQ = 10
SAVE_FREQ = 50
max_lr = 6e-4
min_lr = max_lr * 0.1



# *** MODEL ***
VOCAB_SIZE = 30_000
N_BLOCK = 12
N_HEAD = 12
N_EMBD = 768

# *** OTHER ***
torch.set_float32_matmul_precision('high')
