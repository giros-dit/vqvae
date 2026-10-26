import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

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
Configuración de canales de entrada (CAEZ = 1, I/Q de GIROS/FAWN = 2, CIFAR = 3)
"""
in_channels = utils.get_in_channels(args.dataset)
print(f"--> Dataset: {args.dataset} | Canales de entrada: {in_channels}")

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
    steps_per_epoch = len(training_loader)
    print(f"--> {args.n_updates} updates | {steps_per_epoch} updates por época "
          f"(≈ {args.n_updates / steps_per_epoch:.2f} épocas)")

    pbar = tqdm(range(args.n_updates), desc="Entrenando", unit="upd", dynamic_ncols=True)

    for i in pbar:
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

        if i % 10 == 0:
            pbar.set_postfix(
                epoca=f"{i / steps_per_epoch:.2f}",
                recon=f"{np.mean(results['recon_errors'][-10:]):.4f}",
                perp=f"{np.mean(results['perplexities'][-10:]):.1f}",
            )

        if i % args.log_interval == 0:
            if args.save:
                hyperparameters = args.__dict__
                utils.save_model_and_results(
                    model, results, hyperparameters, args.filename
                )

            tqdm.write(
                f"Update #{i} ({100 * (i + 1) / args.n_updates:.1f} %) | "
                f"Recon Error: {np.mean(results['recon_errors'][-args.log_interval:]):.5f} | "
                f"Loss: {np.mean(results['loss_vals'][-args.log_interval:]):.5f} | "
                f"Perplexity: {np.mean(results['perplexities'][-args.log_interval:]):.1f}"
            )

    pbar.close()

    if args.save:
        utils.save_model_and_results(model, results, args.__dict__, args.filename)
        print(f"--> Modelo final guardado en: {args.filename}")


if __name__ == "__main__":
    train()