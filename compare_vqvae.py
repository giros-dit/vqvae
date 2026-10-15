#!/usr/bin/env python3
"""Script to view and compare original vs VQ-VAE reconstructed .pt tensor files."""

import os
import sys
import matplotlib.pyplot as plt
import torch


def plot_comparison(orig_path, recon_path, save_path=None):

  t_orig = torch.load(orig_path, weights_only=False).to(torch.float64)
  t_recon = torch.load(recon_path, weights_only=False).to(torch.float64)

  orig_img = t_orig[0].numpy()
  recon_img = t_recon[0].numpy()

  vmin = min(orig_img.min(), recon_img.min())
  vmax = max(orig_img.max(), recon_img.max())

  fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, sharex=True)

  im0 = axes[0].imshow(
      orig_img,
      interpolation="nearest",
      aspect="auto",
      cmap="viridis",
      vmin=vmin,
      vmax=vmax,
  )
  axes[0].set_title(f"Original\n({os.path.basename(orig_path)})")
  axes[0].set_ylabel("Subcarrier")
  axes[0].set_xlabel("Symbol")


  im1 = axes[1].imshow(
      recon_img,
      interpolation="nearest",
      aspect="auto",
      cmap="viridis",
      vmin=vmin,
      vmax=vmax,
  )
  axes[1].set_title(f"VQ-VAE Reconstruido\n({os.path.basename(recon_path)})")
  axes[1].set_xlabel("Symbol")

  fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.8)
  plt.tight_layout()

  if save_path:
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to: {save_path}")
  else:
    plt.show()

  plt.close()


if __name__ == "__main__":
  if len(sys.argv) < 6:
    print(
        "Usage: python compare_vqvae.py joined_dir tensored_grids dtype"
        " grid_name output_image.pdf"
    )
    sys.exit(1)

  joined_dir = sys.argv[1]
  tensored_grids = sys.argv[2]
  dtype_in = sys.argv[3]
  grid_name = sys.argv[4]
  output = sys.argv[5]


  if grid_name.endswith(".pt"):
    grid_name = grid_name[:-3]

  orig_path = os.path.join(tensored_grids, dtype_in, f"{grid_name}.pt")
  recon_path = os.path.join(joined_dir, dtype_in, f"{grid_name}.pt")

  if not os.path.exists(orig_path):
    print(f"Error: No existe el original en {orig_path}")
    sys.exit(1)
  if not os.path.exists(recon_path):
    print(f"Error: No existe el reconstruido en {recon_path}")
    sys.exit(1)

  plot_comparison(orig_path, recon_path, output)