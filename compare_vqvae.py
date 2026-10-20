#!/usr/bin/env python3
"""Script to view and compare original vs VQ-VAE reconstructed .pt tensor files."""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_comparison(orig_path, recon_path, save_path=None, mode="magnitude"):
  """Compara un tensor original frente al reconstruido.

  mode: 'magnitude' (|I + jQ|), 'real' (canal 0), o 'imag' (canal 1).
  """
  t_orig = torch.load(orig_path, weights_only=False).float()
  t_recon = torch.load(recon_path, weights_only=False).float()

  # Métricas cuantitativas
  mse = torch.mean((t_orig - t_recon) ** 2).item()
  power = torch.mean(t_orig**2).item()
  nmse_db = (
      10 * np.log10(mse / power) if (power > 0 and mse > 0) else float("nan")
  )

  # Selección de visualización
  if mode == "magnitude" and t_orig.shape[0] == 2:
    orig_img = torch.sqrt(t_orig[0] ** 2 + t_orig[1] ** 2).numpy()
    recon_img = torch.sqrt(t_recon[0] ** 2 + t_recon[1] ** 2).numpy()
    title_suffix = "Magnitud |I + jQ|"
  elif mode == "imag" and t_orig.shape[0] == 2:
    orig_img = t_orig[1].numpy()
    recon_img = t_recon[1].numpy()
    title_suffix = "Canal Imaginario (Q)"
  else:
    orig_img = t_orig[0].numpy()
    recon_img = t_recon[0].numpy()
    title_suffix = "Canal Real (I)"

  # Escala de color compartida
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
  axes[0].set_title(
      f"Original ({title_suffix})\n{os.path.basename(orig_path)}", fontsize=10
  )
  axes[0].set_ylabel("Subcarrier (Frecuencia)")
  axes[0].set_xlabel("Symbol (Tiempo)")

  im1 = axes[1].imshow(
      recon_img,
      interpolation="nearest",
      aspect="auto",
      cmap="viridis",
      vmin=vmin,
      vmax=vmax,
  )
  axes[1].set_title(
      f"Reconstruido (NMSE: {nmse_db:.2f} dB | MSE: {mse:.4e})\n{os.path.basename(recon_path)}",
      fontsize=10,
  )
  axes[1].set_xlabel("Symbol (Tiempo)")

  fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.8)

  if save_path:
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"--> Visualización guardada en: {save_path}")
    print(f"--> NMSE: {nmse_db:.2f} dB | MSE: {mse:.6e}")
  else:
    plt.show()

  plt.close()


def parse_args():
  parser = argparse.ArgumentParser(
      description="Comparar rejillas originales vs reconstruidas VQ-VAE"
  )
  parser.add_argument(
      "--orig",
      type=str,
      required=True,
      help="Ruta directa al archivo .pt original",
  )
  parser.add_argument(
      "--recon",
      type=str,
      required=True,
      help="Ruta directa al archivo .pt reconstruido",
  )
  parser.add_argument(
      "--output",
      type=str,
      default="comparacion.pdf",
      help="Ruta donde guardar la gráfica (PDF o PNG)",
  )
  parser.add_argument(
      "--mode",
      type=str,
      default="magnitude",
      choices=["magnitude", "real", "imag"],
      help="Qué componente mostrar",
  )
  return parser.parse_args()


if __name__ == "__main__":
  args = parse_args()

  if not os.path.exists(args.orig):
    print(f"Error: No existe el original en: {args.orig}")
    sys.exit(1)
  if not os.path.exists(args.recon):
    print(f"Error: No existe el reconstruido en: {args.recon}")
    sys.exit(1)

  plot_comparison(args.orig, args.recon, save_path=args.output, mode=args.mode)