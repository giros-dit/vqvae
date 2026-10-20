
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.residual import ResidualStack


class Encoder(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta 
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Encoder, self).__init__()

        kernel = (4, 1)
        stride = (2, 1)
        padding = (1, 0)

        self.conv_stack = nn.Sequential(
            nn.Conv2d(
                in_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                h_dim, h_dim, kernel_size=(3, 1), stride=(1, 1), padding=padding
            ),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
        )

    def forward(self, x):
        return self.conv_stack(x)


if __name__ == "__main__":
    x = torch.randn(4, 2, 360, 6)

    encoder = Encoder(in_dim=2, h_dim=64, n_res_layers=2, res_h_dim=32)
    encoder_out = encoder(x)
    print("Encoder out shape:", encoder_out.shape)
