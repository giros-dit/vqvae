#!/usr/bin/env python3
"""Script para comparar un tensor original vs reconstruido VQ-VAE en CPU."""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from config import CAEZ_ROOT_DIR, CAEZ_RECON_DIR


def plot_comparison(orig_path, recon_path, save_path=None):
  # Cargar forzando CPU y float32
  t_orig = torch.load(
      orig_path, map_location='cpu', weights_only=False
  ).float()
  t_recon = torch.load(
      recon_path, map_location='cpu', weights_only=False
  ).float()

  # Normalizar a (C, H, W): quitar batch (1, C, H, W) o añadir canal a (H, W)
  if t_orig.ndim == 4:
    t_orig = t_orig.squeeze(0)
  if t_recon.ndim == 4:
    t_recon = t_recon.squeeze(0)
  if t_orig.ndim == 2:
    t_orig = t_orig.unsqueeze(0)
  if t_recon.ndim == 2:
    t_recon = t_recon.unsqueeze(0)

  # Por si alguno conserva filas de padding: comparar solo la zona común
  h = min(t_orig.shape[-2], t_recon.shape[-2])
  w = min(t_orig.shape[-1], t_recon.shape[-1])
  t_orig = t_orig[:, :h, :w]
  t_recon = t_recon[:, :h, :w]

  # Cálculo del NMSE
  mse = torch.mean((t_orig - t_recon) ** 2).item()
  power = torch.mean(t_orig**2).item()
  nmse_db = (
      10 * np.log10(mse / power) if (power > 0 and mse > 0) else float('nan')
  )

  if t_orig.shape[0] == 2:
    # I/Q (FAWN, GIROS): magnitud compleja |I + jQ| = sqrt(I^2 + Q^2)
    orig_img = torch.sqrt(t_orig[0] ** 2 + t_orig[1] ** 2).numpy()
    recon_img = torch.sqrt(t_recon[0] ** 2 + t_recon[1] ** 2).numpy()
    value_label = '|I + jQ|'
  else:
    # Un solo canal (CAEZ): se muestra tal cual
    orig_img = t_orig[0].numpy()
    recon_img = t_recon[0].numpy()
    value_label = 'canal 0'

  vmin = min(orig_img.min(), recon_img.min())
  vmax = max(orig_img.max(), recon_img.max())

  fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, sharex=True)

  im0 = axes[0].imshow(
      orig_img,
      interpolation='nearest',
      aspect='auto',
      cmap='viridis',
      vmin=vmin,
      vmax=vmax,
  )
  axes[0].set_title(
      f'Original ({value_label})\n{os.path.basename(orig_path)}', fontsize=9
  )
  axes[0].set_ylabel(f'Subcarrier (0 a {h - 1})')
  axes[0].set_xlabel(f'Symbol (0 a {w - 1})')

  im1 = axes[1].imshow(
      recon_img,
      interpolation='nearest',
      aspect='auto',
      cmap='viridis',
      vmin=vmin,
      vmax=vmax,
  )
  axes[1].set_title(
      f'Reconstruido (NMSE: {nmse_db:.2f} dB)\n{os.path.basename(recon_path)}',
      fontsize=9,
  )
  axes[1].set_xlabel(f'Symbol (0 a {w - 1})')

  fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.8)

  print(f'Muestra: {os.path.basename(orig_path)}')
  print(f'NMSE: {nmse_db:.2f} dB | MSE: {mse:.6e}')

  if save_path:
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'--> Imagen guardada en: {save_path}')
  else:
    plt.tight_layout()
    plt.show()

  plt.close()


def main():
  parser = argparse.ArgumentParser(
      description='Comparar tensor original vs reconstruido'
  )
  parser.add_argument(
      '--orig_dir',
      type=str,
      default=CAEZ_ROOT_DIR,
      help='Directorio con los tensores originales',
  )
  parser.add_argument(
      '--recon_dir',
      type=str,
      default=os.path.join(CAEZ_RECON_DIR, 'torch.float8_e4m3fn_512'),
      help='Directorio con los tensores reconstruidos',
  )
  parser.add_argument(
      '--grid_name',
      type=str,
      required=True,
      help='Nombre del grid (ej. Path01-01-(6;7)-Robot-0-ssb)',
  )
  parser.add_argument(
      '--output',
      type=str,
      default=None,
      help='Ruta de salida para la imagen (si no se indica, abre ventana interactiva)',
  )

  args = parser.parse_args()

  name = (
      args.grid_name[:-3] if args.grid_name.endswith('.pt') else args.grid_name
  )
  orig_file = os.path.join(args.orig_dir, f'{name}.pt')
  recon_file = os.path.join(args.recon_dir, f'{name}.pt')

  if not os.path.isfile(orig_file):
    print(f'Error: No existe el original en: {orig_file}')
    return
  if not os.path.isfile(recon_file):
    print(f'Error: No existe el reconstruido en: {recon_file}')
    return

  plot_comparison(orig_file, recon_file, save_path=args.output)


if __name__ == '__main__':
  main()