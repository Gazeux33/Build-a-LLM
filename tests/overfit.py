

from model import GPT
from config import *
from Dataloader import DataLoader
from optimizer import AdamW

model = GPT().to(DEVICE)
dataloader = DataLoader(batch_size,block_size,"train","/media/SSD2/LLM/data/test_tokens",loop=False)
optimizer = AdamW(model.parameters(), learning_rate=0.01)

x, y = dataloader.next_batch()
x, y = x.to(DEVICE), y.to(DEVICE)


for i in range(1000):
    optimizer.zero_grad()
    out ,loss = model(x,y)
    loss.backward()
    optimizer.step()
    print(f"iter:{i} loss:{loss.item()}")




