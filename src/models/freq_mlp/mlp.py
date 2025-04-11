import torch
from torch import nn



class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim+64,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        #print(hidden_dim)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim+64, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)


    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """
        #print(input_data.shape)
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        #m = torch.sigmoid(self.fc2(self.drop(self.act(self.fc1(input_data)))))
        #noise = self.noise(hidden.size()).to(device)
        #print(hidden.shape)

        hidden = hidden + input_data # residual
        # hidden = m * hidden + (1 - m)
        return hidden

import torch
import torch.nn as nn
import torch.nn.functional as F

class DropoutUncertain(nn.Module):
    def __init__(self, rate, noise_shape=None, seed=None):
        super(DropoutUncertain, self).__init__()
        self.rate = min(1.0, max(0.0, rate))
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True  # 不确定这个属性在PyTorch中是否需要

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return None
        else:
            return self.noise_shape

    def forward(self, inputs):
        if 0.0 < self.rate < 1.0:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs():
                if noise_shape is not None:
                    dropout_mask = torch.bernoulli(torch.ones(noise_shape) * (1 - self.rate))
                else:
                    dropout_mask = torch.bernoulli(torch.ones_like(inputs) * (1 - self.rate))
                return inputs * dropout_mask / (1 - self.rate)

            return dropped_inputs() if self.training else inputs
        return inputs

    import torch.nn as nn


