import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
import torch.nn.functional as F
import sys
from config import GIROS_ROOT_DIR, GIROS_LABELS_CSV
from datasets.block import BlockDataset, LatentBlockDataset
import numpy as np
sys.path.append("/home/alfonso.inigo/Documentos/vqvae/isac-sensing-presence-ml")
from grid_tensor_dataset import GridTensorDataset


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


# La otra opción potente sería replicar los 2 ultimos simbolos y ponerlos al final, ahora es por si tenemos algún dato que está mal.

def pad_collate_fn(batch):
    tensors, labels = zip(*batch)
    target_width = 14
    
    padded_tensors = []
    for t in tensors:
        
        w = t.shape[-1]
        if w < target_width:
            diff = target_width - w
            pad_left = diff // 2
            pad_right = diff - pad_left
            t = F.pad(t, (pad_left, pad_right), mode='replicate')
            
        elif w > target_width:
            t = t[:, :, :target_width]
            
        padded_tensors.append(t)
        
    return torch.stack(padded_tensors, dim=0), torch.tensor(labels)






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


def load_data_and_data_loaders(dataset, batch_size):
    if dataset == 'CIFAR10':
        training_data, validation_data = load_cifar()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.train_data / 255.0)



    elif dataset == 'BLOCK':
        training_data, validation_data = load_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data / 255.0)


    elif dataset == 'LATENT_BLOCK':
        training_data, validation_data = load_latent_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data)

# Aqui tenemos la carga de datos para nuestro tipo de dato, es decir, aquellos que pertenezcan al grupo GIROS.

    elif 'GIROS' in dataset:
        
        root_dir=GIROS_ROOT_DIR
        labels_csv_path= GIROS_LABELS_CSV
        pin_memory = torch.cuda.is_available()

        dataset = GridTensorDataset(root_dir, labels_csv_path)
    
        training_data, validation_data, test_dataset = torch.utils.data.dataset.random_split(
            dataset, [0.8, 0.1, 0.1]
        )

      
        training_loader = DataLoader(
            training_data,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=pin_memory,
            shuffle=True,
            collate_fn= pad_collate_fn
        )
        validation_loader = DataLoader(
            validation_data,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=pin_memory,
            shuffle=False,
            collate_fn= pad_collate_fn

        )
   
        variances = []
        for x, _ in training_loader:
            variances.append(torch.var(x, unbiased=False))
        x_train_var = torch.mean(torch.stack(variances)).item()
  


        


    else:
        raise ValueError(
            'Invalid dataset: only CIFAR10 and BLOCK datasets are supported.')

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
    
    # Si filename es una ruta absoluta o termina en .pth, úsala directamente
    if filename and (os.path.isabs(filename) or filename.endswith('.pth')):
        save_path = filename
    else:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        base_dir = os.path.join(os.getcwd(), 'results')
        save_path = os.path.join(base_dir, f"{filename}_{timestamp}.pth" if filename else f"vqvae_{timestamp}.pth")

    # Crea automáticamente todas las carpetas intermedias si no existen
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Guarda el archivo
    torch.save(results_to_save, save_path)
