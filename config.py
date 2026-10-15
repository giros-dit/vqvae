
import os

# Ruta base del almacenamiento de datos (cambia solo esta si te mueves de máquina)
# DATA_BASE_DIR = "/data"
# Si en local usas la otra:
DATA_BASE_DIR = "/home/alfonso.inigo/disco6g-data"

# Rutas dependientes
GIROS_ROOT_DIR = os.path.join(DATA_BASE_DIR, "dataset_12-03/tensored_grid/torch.float32")
GIROS_LABELS_CSV = os.path.join(DATA_BASE_DIR, "dataset_12-03/detection_log.csv")