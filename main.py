import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.vqvae import VQVAE
import utils

parser = argparse.ArgumentParser()

"""
Hyperparameters
"""
timestamp = utils.readable_timestamp()

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_updates", type=int, default=5000)
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=32)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--n_embeddings", type=int, default=512)
parser.add_argument("--beta", type=float, default=0.25)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--dataset", type=str, default="CIFAR10")

# Guardado del modelo
parser.add_argument("-save", action="store_true")
parser.add_argument("--filename", type=str, default=timestamp)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.save:
    print(f"--> Modelo y resultados se guardarán en: {args.filename}")

"""
Load data and define batch data loaders
"""
training_data, validation_data, training_loader, validation_loader, x_train_var = (
    utils.load_data_and_data_loaders(args.dataset, args.batch_size)
)

"""
Configuración de canales de entrada (I/Q = 2 para OFDM)
"""
if "GIROS" in args.dataset or "FAWN" in args.dataset:
    in_channels = 2
else:
    in_channels = 3

"""
Set up VQ-VAE model
"""
model = VQVAE(
    args.n_hiddens,
    args.n_residual_hiddens,
    args.n_residual_layers,
    args.n_embeddings,
    args.embedding_dim,
    args.beta,
    in_dim=in_channels,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
model.train()

results = {
    "n_updates": 0,
    "recon_errors": [],
    "loss_vals": [],
    "perplexities": [],
}


def train():
    train_iter = iter(training_loader)

    for i in range(args.n_updates):
        try:
            (x, _) = next(train_iter)
        except StopIteration:
            train_iter = iter(training_loader)
            (x, _) = next(train_iter)

        x = x.to(device)
        optimizer.zero_grad()

        embedding_loss, x_hat, perplexity = model(x)
        recon_loss = torch.mean((x_hat - x) ** 2) / x_train_var
        loss = recon_loss + embedding_loss

        loss.backward()
        optimizer.step()

        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["perplexities"].append(perplexity.cpu().detach().numpy())
        results["loss_vals"].append(loss.cpu().detach().numpy())
        results["n_updates"] = i

        if i % args.log_interval == 0:
            if args.save:
                hyperparameters = args.__dict__
                utils.save_model_and_results(
                    model, results, hyperparameters, args.filename
                )

            print(
                "Update #",
                i,
                "Recon Error:",
                np.mean(results["recon_errors"][-args.log_interval :]),
                "Loss:",
                np.mean(results["loss_vals"][-args.log_interval :]),
                "Perplexity:",
                np.mean(results["perplexities"][-args.log_interval :]),
            )


if __name__ == "__main__":
    train()