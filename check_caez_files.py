"""
Inspecciona el dataset CAEZ: cuántos ficheros son tensores de PyTorch y cuántos
arrays de NumPy, con qué formas y dtypes, y si la escala de los valores coincide.

Uso:
  python check_caez_files.py --dataset_path <carpeta> [--n 5000]
  (--n 0 revisa todos los ficheros; con 515k tarda bastante)
"""
import argparse
import glob
import os
import random
from collections import Counter

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--n', type=int, default=5000, help='Ficheros a revisar (0 = todos)')
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.dataset_path, '*.pt')))
    print(f"Ficheros totales: {len(files)}")
    if args.n and args.n < len(files):
        random.seed(0)
        files = random.sample(files, args.n)
        print(f"Revisando una muestra aleatoria de {len(files)}")

    kinds = Counter()
    stats = {}  # (tipo, dtype, forma) -> [n, suma |x|, max |x|, ejemplo]

    for f in files:
        obj = torch.load(f, weights_only=False)
        if isinstance(obj, torch.Tensor):
            kind, dtype, shape = 'torch', str(obj.dtype), tuple(obj.shape)
            vals = obj.to(torch.float32).abs()
            mean_abs, max_abs = vals.mean().item(), vals.max().item()
        elif isinstance(obj, np.ndarray):
            kind, dtype, shape = 'numpy', str(obj.dtype), tuple(obj.shape)
            vals = np.abs(obj.astype(np.float32))
            mean_abs, max_abs = float(vals.mean()), float(vals.max())
        else:
            kind, dtype, shape = type(obj).__name__, '-', '-'
            mean_abs, max_abs = float('nan'), float('nan')

        kinds[kind] += 1
        key = (kind, dtype, shape)
        if key not in stats:
            stats[key] = [0, 0.0, 0.0, os.path.basename(f)]
        s = stats[key]
        s[0] += 1
        s[1] += mean_abs
        s[2] = max(s[2], max_abs)

    print("\nPor tipo:")
    for k, v in kinds.items():
        print(f"  {k}: {v} ({100 * v / len(files):.1f} %)")

    print("\nPor (tipo, dtype, forma):")
    for (kind, dtype, shape), (n, sum_abs, max_abs, ej) in sorted(stats.items(), key=lambda x: -x[1][0]):
        print(f"  {kind:6s} {dtype:22s} {str(shape):14s} n={n:6d} | "
              f"media|x|={sum_abs / n:.4g} | max|x|={max_abs:.4g} | ej: {ej}")


if __name__ == '__main__':
    main()