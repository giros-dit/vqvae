import argparse
import glob
import os
import torch
import torch.nn.functional as F
from models.vqvae import VQVAE
from config import DATA_BASE_DIR

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compresión y descompresión VQ-VAE con soporte float8"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default= os.path.join(DATA_BASE_DIR,
        "datasets/GIROS/dataset_12-03/vqvae-models/torch.float32/nembeddings512/model2.pth"),
        help="Ruta al archivo .pth del modelo",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default= os.path.join(DATA_BASE_DIR,
        "datasets/GIROS/dataset_12-03/tensored_grid/torch.float8_e4m3fn"),
        help="Carpeta con los tensores originales en float8",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(DATA_BASE_DIR, 
        "datasets/GIROS/dataset_12-03/vqvaed/torch.float8_e4m3fn"),
        help="Carpeta destino para guardar los tensores reconstruidos",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Dispositivo: {device}")
    print(f"--> Modelo: {args.model_path}")
    print(f"--> Entrada: {args.input_dir}")
    print(f"--> Salida: {args.output_dir}")

    checkpoint = torch.load(
        args.model_path, map_location=device, weights_only=False
    )
    hp = checkpoint.get("hyperparameters", {})

    model = VQVAE(
        hp.get("n_hiddens", 128),
        hp.get("n_residual_hiddens", 32),
        hp.get("n_residual_layers", 2),
        hp.get("n_embeddings", 2),
        hp.get("embedding_dim", 64),
        hp.get("beta", 0.25),
        in_dim=2,
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()
    print("[OK] Modelo instanciado y pesos cargados correctamente.")

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.pt")))
    total_files = len(files)
    print(f"--> Procesando {total_files} tensores...")

    with torch.no_grad():
        for idx, f in enumerate(files):
            tensor = torch.load(f, weights_only=False)

            x = tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
            w_orig = x.shape[-1]

            if w_orig < 14:
                x_in = F.pad(x, (0, 14 - w_orig, 0, 0), mode="replicate")
            else:
                x_in = x

            _, x_hat, _ = model(x_in)

            if w_orig < 14:
                x_hat = x_hat[..., :w_orig]

            out_tensor = x_hat.squeeze(0).to(torch.float8_e4m3fn).cpu()

            out_path = os.path.join(args.output_dir, os.path.basename(f))
            torch.save(out_tensor, out_path)

            if (idx + 1) % 500 == 0 or (idx + 1) == total_files:
                print(f"Progreso: {idx + 1}/{total_files} grids procesados...")

    print(
        f"\n[ÉXITO] Todos los grids procesados y guardados en:\n{args.output_dir}"
    )


if __name__ == "__main__":
    main()
