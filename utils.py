import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
import torch.nn.functional as F
import sys
from config import FAWN_ROOT_DIR, CAEZ_ROOT_DIR
from datasets.block import BlockDataset, LatentBlockDataset
import numpy as np
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "isac-sensing-presence-ml"))
try:
    from grid_tensor_dataset import GridTensorDataset
except ImportError:
    GridTensorDataset = None
from grid_tensor_fawn_dataset import FAWNGridSymbolTensorDataset
from grid_tensor_caez_dataset import CAEZGridTensorDataset
from config import FAWN_ROOT_DIR, CAEZ_ROOT_DIR, FRAUNHOFER_ROOT_DIR
from grid_tensor_fraunhofer_dataset import FraunhoferTensorDataset, FRAUNHOFER_TAPS

def get_in_channels(dataset_name):
    """Número de canales de entrada según el dataset."""
    if "CAEZ" in dataset_name or "FRAUNHOFER" in dataset_name:
        return 1
    if "GIROS" in dataset_name or "FAWN" in dataset_name:
        return 2
    return 3


def load_cifar():
    train = datasets.CIFAR10(root="data", train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))

    val = datasets.CIFAR10(root="data", train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
    return train, val


def load_block():
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
        '/data/randact_traj_length_100_n_trials_1000_n_contexts_1.npy'

    train = BlockDataset(data_file_path, train=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ]))

    val = BlockDataset(data_file_path, train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(
                               (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ]))
    return train, val

def load_latent_block():
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
        '/data/latent_e_indices.npy'

    train = LatentBlockDataset(data_file_path, train=True,
                         transform=None)

    val = LatentBlockDataset(data_file_path, train=False,
                       transform=None)
    return train, val


def pad_collate_fn(batch):
    """
    Collate de FAWN: asegura 6 símbolos OFDM por grid.
    Si un grid tiene menos de 6 símbolos, se rellena por los dos lados replicando
    los bordes; si tiene más, se recorta. Es por si algún dato viene mal.
    Otra opción sería replicar los 2 últimos símbolos y ponerlos al final.
    """
    tensors, labels = zip(*batch)
    target_width = 6

    padded_tensors = []
    for t in tensors:
        t = t.float()

        if t.ndim == 3:
            w = t.shape[-1]
            if w < target_width:
                diff = target_width - w
                pad_left = diff // 2
                pad_right = diff - pad_left
                t = F.pad(t, (pad_left, pad_right), mode="replicate")
            elif w > target_width:
                t = t[:, :, :target_width]

        padded_tensors.append(t)

    return torch.stack(padded_tensors, dim=0), list(labels)


def caez_collate_fn(batch):
    """CAEZ: el dataset ya entrega (1, 276, 10); solo se apila y se dejan los nombres como lista."""
    tensors, labels = zip(*batch)
    return torch.stack([t.float() for t in tensors], dim=0), list(labels)





def data_loaders(train_data, val_data, batch_size):

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True)
    return train_loader, val_loader


def load_data_and_data_loaders(dataset_name, batch_size):
    if dataset_name == 'CIFAR10':
        training_data, validation_data = load_cifar()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.data / 255.0)

    elif dataset_name == 'BLOCK':
        training_data, validation_data = load_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.data / 255.0)

    elif dataset_name == 'LATENT_BLOCK':
        training_data, validation_data = load_latent_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.data)

    elif "FAWN" in dataset_name:
        root_dir = FAWN_ROOT_DIR
        pin_memory = torch.cuda.is_available()

        fawn_dataset = FAWNGridSymbolTensorDataset(root_dir)

        training_data, validation_data, test_dataset = (
            torch.utils.data.random_split(
                fawn_dataset,
                [0.8, 0.1, 0.1],
                generator=torch.Generator().manual_seed(42),
            )
        )

        training_loader = DataLoader(
            training_data,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=pin_memory,
            shuffle=True,
            collate_fn=pad_collate_fn,
        )
        validation_loader = DataLoader(
            validation_data,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=pin_memory,
            shuffle=False,
            collate_fn=pad_collate_fn,
        )

        variances = []
        with torch.no_grad():
            for x, _ in training_loader:
                variances.append(torch.var(x.float(), unbiased=False).item())
        x_train_var = float(np.mean(variances))

    elif "FRAUNHOFER" in dataset_name:
        pin_memory = torch.cuda.is_available()
        loader_workers = 2

        
        fr_dataset = FraunhoferTensorDataset(FRAUNHOFER_ROOT_DIR)

        
        training_data, validation_data, test_dataset = (
            torch.utils.data.random_split(
                fr_dataset,
                [0.8, 0.1, 0.1],
                generator=torch.Generator().manual_seed(42),
            )
        )

        
        training_loader = DataLoader(
            training_data, batch_size=batch_size, num_workers=loader_workers,
            pin_memory=pin_memory, shuffle=True, collate_fn=caez_collate_fn,
        )
        validation_loader = DataLoader(
            validation_data, batch_size=batch_size, num_workers=loader_workers,
            pin_memory=pin_memory, shuffle=False, collate_fn=caez_collate_fn,
        )

       
        var_loader = DataLoader(
            training_data, batch_size=512, num_workers=loader_workers,
            shuffle=False, collate_fn=caez_collate_fn,
        )
        total_sum, total_sq, total_n = 0.0, 0.0, 0
        with torch.no_grad():
            for x, _ in tqdm(var_loader, desc="Calculando varianza", unit="lote", dynamic_ncols=True):
                x = x[:, :, :FRAUNHOFER_TAPS, :].double()
                total_sum += x.sum().item()
                total_sq += (x ** 2).sum().item()
                total_n += x.numel()
        mean = total_sum / total_n
        x_train_var = float(total_sq / total_n - mean ** 2)
        print(f"--> Varianza de train ({len(training_data)} muestras): {x_train_var:.6e}")

    elif "CAEZ" in dataset_name:
        pin_memory = torch.cuda.is_available()

        
        caez_dataset = CAEZGridTensorDataset(CAEZ_ROOT_DIR)
        loader_workers = 2

        
        training_data, validation_data, test_dataset = (
            torch.utils.data.random_split(
                caez_dataset,
                [0.8, 0.1, 0.1],
                generator=torch.Generator().manual_seed(42),
            )
        )

        training_loader = DataLoader(
            training_data,
            batch_size=batch_size,
            num_workers=loader_workers,
            pin_memory=pin_memory,
            shuffle=True,
            collate_fn=caez_collate_fn,
        )
        validation_loader = DataLoader(
            validation_data,
            batch_size=batch_size,
            num_workers=loader_workers,
            pin_memory=pin_memory,
            shuffle=False,
            collate_fn=caez_collate_fn,
        )

       
        n_var = min(20000, len(training_data))
        var_idx = torch.randperm(
            len(training_data), generator=torch.Generator().manual_seed(0)
        )[:n_var].tolist()
        var_loader = DataLoader(
            torch.utils.data.Subset(training_data, var_idx),
            batch_size=256,
            num_workers=loader_workers,
            shuffle=False,
            collate_fn=caez_collate_fn,
        )
        total_sum, total_sq, total_n = 0.0, 0.0, 0
        with torch.no_grad():
            for x, _ in tqdm(var_loader, desc="Calculando varianza", unit="lote", dynamic_ncols=True):
                x = x[:, :, :273, :].double()
                total_sum += x.sum().item()
                total_sq += (x ** 2).sum().item()
                total_n += x.numel()
        mean = total_sum / total_n
        x_train_var = float(total_sq / total_n - mean ** 2)
        print(f"--> Varianza de train (estimada con {n_var} grids): {x_train_var:.6e}")

    else:
        raise ValueError(
            f'Invalid dataset: {dataset_name} is not supported.')

    return training_data, validation_data, training_loader, validation_loader, x_train_var


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()


def save_model_and_results(model, results, hyperparameters, filename):
    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    
    if filename and (os.path.isabs(filename) or filename.endswith('.pth')):
        save_path = filename
    else:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        base_dir = os.path.join(os.getcwd(), 'results')
        save_path = os.path.join(base_dir, f"{filename}_{timestamp}.pth" if filename else f"vqvae_{timestamp}.pth")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(results_to_save, save_path)