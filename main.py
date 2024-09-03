from train import Trainer

if __name__ == "__main__":
    trainer = Trainer(compile_model=True, load=True)
    trainer.train()
