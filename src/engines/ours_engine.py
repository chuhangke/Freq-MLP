import torch
import numpy as np
from src.base.engine import BaseEngine
from src.utils.metrics import masked_mape, masked_rmse
import math
class OURS_Engine(BaseEngine):
    def __init__(self,beta, **args):
        self.beta=beta
        
    def train_batch(self):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        self._dataloader['train_loader'].shuffle()
        for X, label in self._dataloader['train_loader'].get_iterator():
            self._optimizer.zero_grad()

            # X (b, t, n, f), label (b, t, n, 1)
            X, label = self._to_device(self._to_tensor([X, label]))
            mu_std,pred = self.model(X, label)
            pred, label = self._inverse_transform([pred, label])
            mu=mu_std[0]
            std=mu_std[1]

            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                print('Check mask value', mask_value)

            loss = self._loss_fn(pred, label, mask_value)
            info_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(0).mean().div(math.log(2))
            beta = self.beta
            loss = loss + beta * info_loss
            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()

            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(loss.item())
            train_mape.append(mape)
            train_rmse.append(rmse)

            self._iter_cnt += 1
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)
