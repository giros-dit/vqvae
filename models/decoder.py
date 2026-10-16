
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.residual import ResidualStack


class Decoder(nn.Module):
      """
    This is the p_phi (x|z) network. Given a latent sample z p_phi 
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    - out_dim : the output channels (default: 3 for RGB, 2 for CSI)
    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, out_dim=2):
        super(Decoder, self).__init__()
        kernel = (4, 1)
        stride = (2, 1)
        padding = (1, 0)

        self.inverse_conv_stack = nn.Sequential(
            
            nn.ConvTranspose2d(
                in_dim, h_dim, kernel_size=(3, 1), stride=(1, 1), padding=padding
            ),
            
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            
            nn.ConvTranspose2d(
                h_dim, h_dim, kernel_size=kernel, stride=stride, padding=padding
            ),
            nn.ReLU(),
            
            nn.ConvTranspose2d(
                h_dim, out_dim, kernel_size=kernel, stride=stride, padding=padding
            ),
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)


if __name__ == "__main__":
    z_q = torch.randn(32, 64, 135, 14)

    decoder = Decoder(
        in_dim=64, h_dim=64, n_res_layers=2, res_h_dim=32, out_dim=2
    )
    decoder_out = decoder(z_q)
    print("Decoder out shape:", decoder_out.shape)
