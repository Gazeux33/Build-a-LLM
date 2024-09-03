import json
import math
import os
import time
from typing import Tuple, Any

import torch
from torch.optim import AdamW
from torch import nn
from tqdm import tqdm

import utils
from config import *
from model import GPT
from Dataloader import DataLoader


class Trainer:
    def __init__(self, compile_model: bool = False, load: bool = True) -> None:
        self.train_dataloader:DataLoader = DataLoader(batch_size, block_size, "train", TOKENS_DIR, loop=False)
        self.test_dataloader: DataLoader = DataLoader(batch_size, block_size, "test", TOKENS_DIR, loop=True)
        self.model:nn.Module = GPT().to(DEVICE)
        self.optimizer = AdamW(self.model.parameters(), lr=max_lr)
        assert total_batch_size % (batch_size * block_size) == 0
        self.gradient_accumulation_steps:int = total_batch_size // (batch_size * block_size)
        self.nb_batches = self.train_dataloader.count_total_batches()
        self.prev_nb_step:int = self.nb_batches // self.gradient_accumulation_steps * EPOCHS
        self.warmup_steps = int(0.1 * self.prev_nb_step)

        self.step:int= 0
        self.n_tokens:int= 0
        self.epochs:int= 1
        self.train_loss =  []
        self.test_loss = []
        self.test_accuracy = []

        if load:
            self._load_model()
            self._load_metrics()
            self._buffer_tokens()
        else:
            self._reset_metrics()

        if compile_model and torch.cuda.is_available():
            print("compiling model...")
            self.model = torch.compile(self.model)
        print(f"Number of batchs: {self.nb_batches}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"Number of steps: {self.prev_nb_step}")


    # WARNING
    def _load_model(self) -> None:
        try:
            self.model = utils.load_weights(self.model, os.path.join(MODEL_DIR,"last.pth"))
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("No saved model found. Starting from scratch.")

    def train(self):
        for epoch in range(EPOCHS):
            while self.train_dataloader:
                start_time = time.time()
                self._train_step(epoch)
                elapsed_time = round((time.time() - start_time) * 1000, 2)
                self._log_state(elapsed_time)
            self.train_dataloader.reset()

    def _train_step(self, epoch: int):
        self.model.train()
        current_lr = self._get_lr(self.step)
        self._set_lr(current_lr)
        loss = None
        self.optimizer.zero_grad()
        for _ in range(self.gradient_accumulation_steps):
            x, y = self.train_dataloader.next_batch()
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, loss = self._predict(x, y)
            loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.step += 1
        self.train_loss.append(loss.item())
        self.n_tokens += block_size * batch_size * self.gradient_accumulation_steps
        self._update_state()

    def _predict(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if device_name == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return self.model(x, y)
        return self.model(x, y)

    def _set_lr(self, lr: float):
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def _get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return max_lr * (step + 1) / self.warmup_steps
        if step > self.prev_nb_step:
            return min_lr

        decay_ratio = (step - self.warmup_steps) / (self.prev_nb_step - self.warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + cosine_decay * (max_lr - min_lr)

    def _update_state(self):
        if self.step % EVAL_FREQ == 0:
            self.eval()
        if self.step % SAVE_FREQ == 0:
            utils.save_model(self.model, "last.pth")
            self._save_metrics()
            print("Model and metrics saved.")

    def _log_state(self, time_elapsed: float):
        #print(f"epoch:{self.epochs}  step:{self.step}/{self.prev_nb_step}  train_loss:{round(self.train_loss[-1],4)}  time:{time_elapsed} ms")
        print(f"epoch:{self.epochs}  step:{self.step}/{self.prev_nb_step}   time:{time_elapsed} ms")

    def eval(self):
        print("Evaluating...")
        self.model.eval()
        x_test, y_test = self.test_dataloader.next_batch()
        x_test, y_test = x_test.to(DEVICE), y_test.to(DEVICE)
        logits, loss = self.model(x_test, y_test)
        accuracy = self.model.accuracy(logits, y_test)
        self.test_loss.append(loss.item())
        self.test_accuracy.append(accuracy.item())


    def _buffer_tokens(self):
        buffer_tokens = 0
        while self.n_tokens > buffer_tokens:
            x, _ = self.train_dataloader.next_batch()
            buffer_tokens += x.size(0) * x.size(1)
        print(f"Tokens buffered: {buffer_tokens}")

    def _save_metrics(self)->None:
        data = self._load_file(MODEL_DIR, "metrics.json")
        obj = {
            "epoch": self.epochs,
            "step": self.step,
            "n_tokens": self.n_tokens,
            "train_loss": data.get("train_loss",[]) + self.train_loss,
            "test_loss": data.get("test_loss",[]) + self.test_loss,
            "test_accuracy" : data.get("test_accuracy",[]) + self.test_accuracy
        }
        with open(os.path.join(MODEL_DIR,"metrics.json"), "w+") as f:
            json.dump(obj, f,indent=2)


        self.train_loss = []
        self.test_loss = []
        self.test_accuracy = []



    @staticmethod
    def _load_file(*path) -> dict[str,Any]:
        path = os.path.join(*path)
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            return data
        else:
            raise FileNotFoundError(f"No {path} found.")

    def _load_metrics(self)->None:
        data = self._load_file(MODEL_DIR, "metrics.json")
        self.n_tokens = data["n_tokens"]
        self.step = data["step"]
        self.epoch = data["epoch"]

    @staticmethod
    def _reset_metrics():
        with open(os.path.join(MODEL_DIR,"metrics.json"), "w+") as f:
            json.dump({}, f,indent=2)











