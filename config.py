
import os

# Ruta base del almacenamiento de datos (cambia solo esta si te mueves de máquina)
# DATA_BASE_DIR = "/data"
# Si en local usas la otra:
DATA_BASE_DIR = "/home/alfonso.inigo/disco6g-data"

# Rutas dependientes
FAWN_ROOT_DIR = os.path.join(DATA_BASE_DIR, "datasets/medidas-fawn/3gpp/tensored/torch.float8_e4m3fn")
#GIROS_LABELS_CSV = os.path.join(DATA_BASE_DIR, "datasets/GIROS/dataset_12-03/detection_log.csv")

CAEZ_ROOT_DIR= os.path.join(DATA_BASE_DIR, "datasets/caez/caez_5g_outdoor/tensored/torch.float8_e4m3fn")