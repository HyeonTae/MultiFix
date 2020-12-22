import numpy as np
import torch

class EarlyStopping:
    def __init__(self, save_path, patience):
        self.save_path = save_path
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.best_score = None
        self.best_epoch = 0
        self.best_f1_score = 0

    def __call__(self, dev_loss, model, epoch, f1_score):

        score = -dev_loss
        if self.patience is None:
            if self.best_f1_score < f1_score:
                self.best_epoch = epoch
                self.best_f1_score = f1_score
                self.save_checkpoint(dev_loss, model)

            if f1_score > 0.9999:
                self.counter += 1
                if self.counter >= 4:
                    self.save_checkpoint(dev_loss, model)
                    self.early_stop = True
            else:
                self.counter = 0
        else:
            if self.best_score is None:
                self.best_score = score
                self.best_epoch = epoch
                self.best_f1_score = f1_score
                self.save_checkpoint(dev_loss, model)
            elif score < self.best_score:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.best_epoch = epoch
                self.best_f1_score = f1_score
                self.save_checkpoint(dev_loss, model)
                self.counter = 0

    def save_checkpoint(self, dev_loss, model):
        torch.save(model.state_dict(), self.save_path)
