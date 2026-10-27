import argparse
import glob
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from config import (
    FAWN_ROOT_DIR, FAWN_MODELS_DIR, FAWN_RECON_DIR,
    CAEZ_ROOT_DIR, CAEZ_MODELS_DIR, CAEZ_RECON_DIR,
)
from models.vqvae import VQVAE
from grid_tensor_caez_dataset import (
    load_grid, list_caez_grid_files,
    CAEZ_CHANNELS, CAEZ_SUBCARRIERS, CAEZ_PADDED_SUBCARRIERS, CAEZ_SYMBOLS,
)

# Configuración por dataset. Las rutas salen de config.py y las dimensiones de
# CAEZ de grid_tensor_caez_dataset.py, para no tenerlas definidas en dos sitios.
#   models_dir -> el modelo de K embeddings está en models_dir/nembeddings{K}/model.pth
#   recon_dir  -> las reconstrucciones van a recon_dir/torch.float8_e4m3fn_{K}/
DATASETS = {
    "FAWN": dict(input_dir=FAWN_ROOT_DIR, models_dir=FAWN_MODELS_DIR, recon_dir=FAWN_RECON_DIR,
                 in_dim=2, height=360, padded_height=360, width=6),
    "CAEZ": dict(input_dir=CAEZ_ROOT_DIR, models_dir=CAEZ_MODELS_DIR, recon_dir=CAEZ_RECON_DIR,
                 in_dim=CAEZ_CHANNELS, height=CAEZ_SUBCARRIERS,
                 padded_height=CAEZ_PADDED_SUBCARRIERS, width=CAEZ_SYMBOLS),
}


def parse_args():
  parser = argparse.ArgumentParser(
      description="Compresión y descompresión VQ-VAE con soporte float8 (FAWN / CAEZ)"
  )
  parser.add_argument("--dataset", type=str, default="CAEZ", choices=DATASETS.keys())
  parser.add_argument(
      "--n_embeddings", type=int, default=512,
      help="Solo se usa para construir las rutas por defecto (nembeddings{K}, _{K})",
  )
  parser.add_argument("--model_path", type=str, default=None,
                      help="Ruta al .pth (por defecto: <models_dir>/nembeddings{K}/model.pth)")
  parser.add_argument("--input_dir", type=str, default=None,
                      help="Tensores originales (por defecto: los de config.py)")
  parser.add_argument("--output_dir", type=str, default=None,
                      help="Destino (por defecto: <recon_dir>/torch.float8_e4m3fn_{K})")
  args = parser.parse_args()

  cfg = DATASETS[args.dataset]
  k = args.n_embeddings
  if args.model_path is None:
    args.model_path = os.path.join(cfg["models_dir"], f"nembeddings{k}", "model.pth")
  if args.input_dir is None:
    args.input_dir = cfg["input_dir"]
  if args.output_dir is None:
    args.output_dir = os.path.join(cfg["recon_dir"], f"torch.float8_e4m3fn_{k}")
  return args


def prepare_input(tensor, cfg):
  """(C,H,W) o (H,W) en float8 -> (1, C, padded_H, W) en float32 + ancho original."""
  x = tensor.to(torch.float32)
  if x.ndim == 2:
    x = x.unsqueeze(0)
  x = x[:, :cfg["height"], :]
  x = x.unsqueeze(0)

  w_orig = x.shape[-1]
  if w_orig < cfg["width"]:
    diff = cfg["width"] - w_orig
    # Mismo relleno centrado que en el entrenamiento
    x = F.pad(x, (diff // 2, diff - diff // 2, 0, 0), mode="replicate")
  elif w_orig > cfg["width"]:
    x = x[..., :cfg["width"]]
    w_orig = cfg["width"]

  pad_h = cfg["padded_height"] - x.shape[-2]
  if pad_h > 0:
    x = F.pad(x, (0, 0, 0, pad_h), mode="constant", value=0.0)
  return x, w_orig


def test_split_files(files, seed=42):
  """Reproduce el 10% de test de random_split([0.8, 0.1, 0.1]) con la misma semilla."""
  _, _, test_idx = torch.utils.data.random_split(
      range(len(files)), [0.8, 0.1, 0.1],
      generator=torch.Generator().manual_seed(seed),
  )
  return {files[i] for i in test_idx}


def nmse_ratio(orig, recon):
  mse = torch.sum((orig - recon) ** 2).item()
  power = torch.sum(orig ** 2).item()
  if power <= 0:
    return None
  return mse / power


def main():
  args = parse_args()
  cfg = DATASETS[args.dataset]

  os.makedirs(args.output_dir, exist_ok=True)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"--> Dispositivo: {device}")
  print(f"--> Dataset: {args.dataset}")
  print(f"--> Modelo: {args.model_path}")
  print(f"--> Entrada: {args.input_dir}")
  print(f"--> Salida: {args.output_dir}")

  if not os.path.isfile(args.model_path):
    raise FileNotFoundError(f"No se encontró el modelo en: {args.model_path}")

  checkpoint = torch.load(
      args.model_path, map_location=device, weights_only=False
  )
  hp = checkpoint.get("hyperparameters", {})

  if "dataset" in hp and args.dataset not in hp["dataset"]:
    print(f"[AVISO] El modelo se entrenó con dataset={hp['dataset']} y estás usando {args.dataset}")

  model = VQVAE(
      hp.get("n_hiddens", 128),
      hp.get("n_residual_hiddens", 32),
      hp.get("n_residual_layers", 2),
      hp.get("n_embeddings", 512),
      hp.get("embedding_dim", 64),
      hp.get("beta", 0.25),
      in_dim=cfg["in_dim"],
      # Modelos antiguos no guardaban este campo -> eran del cuantizador original
      quantizer=hp.get("quantizer", "original"),
  ).to(device)

  model.load_state_dict(checkpoint["model"])
  model.eval()
  print(f"[OK] Modelo cargado (n_embeddings={hp.get('n_embeddings', 512)}, "
        f"cuantizador={hp.get('quantizer', 'original')}).")

  if args.dataset == "CAEZ":
    # Solo los grids H-t_*-B_*-A_*.pt: misma lista (y por tanto mismo split) que en el entrenamiento
    files = list_caez_grid_files(args.input_dir, verbose=True)
  else:
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.pt")))
  total_files = len(files)
  if total_files == 0:
    raise RuntimeError(
        f"No se encontraron archivos .pt en el directorio: {args.input_dir}"
    )

  # Mismo split que en utils.load_data_and_data_loaders (semilla 42)
  test_files = test_split_files(files)

  print(f"--> Procesando {total_files} tensores ({len(test_files)} del test split)...")

  nmse_all, nmse_test = [], []

  with torch.no_grad():
    pbar = tqdm(files, desc="Comprimiendo", unit="grid", dynamic_ncols=True)
    for idx, f in enumerate(pbar):
      # Carga a float32 y falla con mensaje claro si el fichero no es un grid
      tensor = load_grid(f, torch.float32)
      orig_ndim = tensor.ndim

      x_in, w_orig = prepare_input(tensor, cfg)
      _, x_hat, _ = model(x_in.to(device))

      # Quitar el padding: subportadoras reales y ancho original
      x_hat = x_hat[..., :cfg["height"], :]
      if w_orig < cfg["width"]:
        diff = cfg["width"] - w_orig
        left = diff // 2
        x_hat = x_hat[..., left:left + w_orig]

      # Clamping para evitar overflow en el rango dinámico de float8_e4m3fn
      x_clamped = torch.clamp(x_hat.squeeze(0), min=-448.0, max=448.0)
      out_tensor = x_clamped.to(torch.float8_e4m3fn).cpu()
      if orig_ndim == 2:
        out_tensor = out_tensor.squeeze(0)  # se guarda con la misma forma que el original

      out_path = os.path.join(args.output_dir, os.path.basename(f))
      torch.save(out_tensor, out_path)

      # NMSE sobre lo que realmente se guarda (incluye el error de float8)
      orig_crop = tensor.to(torch.float32)
      if orig_crop.ndim == 2:
        orig_crop = orig_crop.unsqueeze(0)
      orig_crop = orig_crop[:, :cfg["height"], :w_orig]
      recon = out_tensor.to(torch.float32)
      if recon.ndim == 2:
        recon = recon.unsqueeze(0)
      ratio = nmse_ratio(orig_crop, recon)
      if ratio is not None:
        nmse_all.append(ratio)
        if f in test_files:
          nmse_test.append(ratio)

      # NMSE acumulado hasta ahora, junto a la barra
      if (idx + 1) % 500 == 0 and nmse_all:
        pbar.set_postfix(nmse_dB=f"{10 * np.log10(np.mean(nmse_all)):.2f}")

  def to_db(values):
    return 10 * np.log10(np.mean(values)) if values else float("nan")

  print(f"\n[ÉXITO] Grids guardados en:\n{args.output_dir}")
  print(f"\nNMSE medio (todos los grids): {to_db(nmse_all):.2f} dB")
  print(f"NMSE medio (solo test split):  {to_db(nmse_test):.2f} dB  <- este es el de la curva")


if __name__ == "__main__":
  main()