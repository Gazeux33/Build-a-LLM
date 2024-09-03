import torch
import numpy as np
import os

from utils import get_tokens_paths


class DataLoader:
    def __init__(self, b: int, t: int, split: str, tokens_dir: str, loop: bool=False) -> None:
        print(f"create dataloader: {split}")
        self.loop = loop
        self.tokens = None
        self.current_position = None
        self.current_file_index = None
        self.data_files = get_tokens_paths(os.path.join(tokens_dir), split)
        if len(self.data_files) == 0:
            raise Exception(f"No data files found for {split}")
        self.B = b
        self.T = t
        self.split = split
        self.reset()

    def reset(self) -> None:
        self.current_file_index = 0
        self.tokens = self.load_tokens(self.data_files[self.current_file_index], log=False)
        self.current_position = 0

    @staticmethod
    def load_tokens(filename: str, log: bool) -> torch.Tensor:
        if log:
            print(f"Loading tokens from: {filename}")
        npt = np.load(filename)
        npt = npt.astype(np.int32)
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        while True:
            if self.current_position + self.B * self.T >= len(self.tokens):
                if self.current_file_index + 1 < len(self.data_files):
                    self.current_file_index += 1
                    self.tokens = self.load_tokens(self.data_files[self.current_file_index], log=True)
                    self.current_position = 0
                elif self.loop:
                    self.reset()

            end_pos = self.current_position + self.B * self.T + 1
            if end_pos <= len(self.tokens):
                x = self.tokens[self.current_position:self.current_position + self.B * self.T]
                y = self.tokens[self.current_position + 1: end_pos]

                x = x.view(self.B, self.T)
                y = y.view(self.B, self.T)

                self.current_position += self.B * self.T
                return x, y

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if not self:
            raise StopIteration
        return self.next_batch()

    def __bool__(self) -> bool:
        if self.loop:
            return True
        enough_tokens_left = (self.current_position + self.B * self.T) < len(self.tokens)
        more_files_to_process = self.current_file_index + 1 < len(self.data_files)
        return enough_tokens_left or more_files_to_process

    def count_total_batches(self) -> int:
        total_tokens = 0
        for file in self.data_files:
            tokens = self.load_tokens(file, log=False)
            total_tokens += len(tokens)
        total_batches = total_tokens // (self.B * self.T)

        return total_batches
