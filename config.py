import os

# Ruta base del almacenamiento de datos, elegida automáticamente:
#   1. Variable de entorno DISCO6G_DATA, si está definida (para forzar una ruta concreta)
#   2. /data, si existe (servidor con GPU)
#   3. /home/alfonso.inigo/disco6g-data (local)
_GPU_DATA_DIR = "/data"
_LOCAL_DATA_DIR = "/home/alfonso.inigo/disco6g-data"

if os.environ.get("DISCO6G_DATA"):
    DATA_BASE_DIR = os.environ["DISCO6G_DATA"]
elif os.path.isdir(os.path.join(_GPU_DATA_DIR, "datasets")):
    DATA_BASE_DIR = _GPU_DATA_DIR
else:
    DATA_BASE_DIR = _LOCAL_DATA_DIR

print(f"--> DATA_BASE_DIR = {DATA_BASE_DIR}")

# Rutas dependientes
FAWN_ROOT_DIR = os.path.join(DATA_BASE_DIR, "datasets/medidas-fawn/3gpp/tensored/torch.float8_e4m3fn")
# FAWN mantiene la estructura que ya tenías: vqvae-models/torch.float32/nembeddings{K}/model.pth
FAWN_MODELS_DIR = os.path.join(DATA_BASE_DIR, "datasets/medidas-fawn/3gpp/vqvae-models/torch.float32")
FAWN_RECON_DIR = os.path.join(DATA_BASE_DIR, "datasets/medidas-fawn/3gpp/vqvaed")

CAEZ_ROOT_DIR = os.path.join(DATA_BASE_DIR, "datasets/caez/caez_5g_outdoor/tensored/torch.float8_e4m3fn")
# Modelos: vqvae-models/nembeddings{K}/model.pth
CAEZ_MODELS_DIR = os.path.join(DATA_BASE_DIR, "datasets/caez/caez_5g_outdoor/vqvae-models")
# Reconstrucciones del compressor: vqvaed/torch.float8_e4m3fn_{K}/
CAEZ_RECON_DIR = os.path.join(DATA_BASE_DIR, "datasets/caez/caez_5g_outdoor/vqvaed")
#GIROS_LABELS_CSV = os.path.join(DATA_BASE_DIR, "datasets/GIROS/dataset_12-03/detection_log.csv")

FRAUNHOFER_ROOT_DIR = os.path.join(DATA_BASE_DIR, "datasets/fraunhofer5G/preprocessed_test_tensored/torch.float8_e4m3fn")
FRAUNHOFER_MODELS_DIR = os.path.join(DATA_BASE_DIR, "datasets/fraunhofer5G/vqvae-models-ema")
FRAUNHOFER_RECON_DIR = os.path.join(DATA_BASE_DIR, "datasets/fraunhofer5G/vqvaed_ema")

GIROS_ROOT_DIR = os.path.join(DATA_BASE_DIR, "datasets/GIROS/dataset_12-03/tensored_grid/torch.float32")
GIROS_MODELS_DIR = os.path.join(DATA_BASE_DIR, "datasets/GIROS/dataset_12-03/vqvae-models-ema")
GIROS_RECON_DIR = os.path.join(DATA_BASE_DIR, "datasets/GIROS/dataset_12-03/vqvaed_ema")