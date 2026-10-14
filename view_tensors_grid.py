#!/usr/bin/env python3
"""
Script to view and visualize .pt tensor files
"""
import torch
import matplotlib.pyplot as plt
import sys
import os
import glob
import matplot2tikz


def plot_tensors(tensor_paths, save_path=None, just_amplitude=True):
    """Load and plot a grid of tensors to a file (just amplitude implemented)"""


    # Obtain the max/min value of tensors
    vmin, vmax = 100000000000000, -1000000000000
    for w, w_tensor_paths in enumerate(tensor_paths):
        for h, tensor_path in enumerate(w_tensor_paths):
            t = torch.tensor(torch.load(tensor_path), dtype=torch.float64)
            if t[0].max() > vmax:
                vmax = t[0].max()
            if t[0].min() < vmin:
                vmin = t[0].min()



    fig, axes = plt.subplots(len(tensor_paths), len(tensor_paths[0]),
        figsize=(14, 6), sharey=True, sharex=True)
    for w, w_tensor_paths in enumerate(tensor_paths):
        for h, tensor_path in enumerate(w_tensor_paths):
            t = torch.tensor(torch.load(tensor_path), dtype=torch.float64)
            W = w
            H = len(w_tensor_paths) - 1 - h
            axes[H][W].imshow(t[0].numpy(), interpolation='nearest',
                aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)

            wl = ['0%', '30%', '60%', '90%'][w]
            hl = ['0%', '30%', '60%', '90%'][h]
            axes[H][W].text(.65, 0.6, f'time={wl}\nfreq={hl}',
                transform=axes[H][W].transAxes,
                bbox=dict(facecolor='white', alpha=1))
            if w == 0:
                axes[H][W].set_ylabel('Subcarrier')
            if h == 0:
                axes[H][W].set_xlabel('Symbol')


    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        #matplot2tikz.save(save_path[:-3] + 'tex', dpi=150)
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()


    # Free memory
    plt.close()




if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python view_tensors_grid.py joined_dir tensored_grids dtype grid_name output_image.pdf")
        # print("\nExample:")
        # print("  python view_tensor.py ../dataset16-02/tensors/grid_20260216_111121_654.pt")
        # print("  python view_tensor.py ../dataset16-02/tensors/grid_20260216_111121_654.pt /tmp/output.pdf")
        sys.exit(1)
    
    joined_dir = sys.argv[1]
    tensored_grids = sys.argv[2]
    dtype_in = sys.argv[3]
    grid_name = sys.argv[4]
    output = sys.argv[5]

    if dtype_in == 'torch.float8_e4m3fn':
        dtype = torch.float8_e4m3fn
    elif dtype_in == 'torch.float16':
        dtype = torch.float16
    elif dtype_in == 'torch.float32':
        dtype = torch.float32
    elif dtype_in == 'torch.float64':
        dtype = torch.float64



    # Matrix of paths to plot the reconstruction
    paths = [
        [
            joined_dir + f'/{dtype_in}-Hcomp-0.{H}_Wcomp-0.{W}/{grid_name}.pt'
            for H in [0,3,6,9]
        ]
        for W in [0,3,6,9]
    ]
    paths[0][0] = tensored_grids + f'/{dtype_in}/{grid_name}.pt'

    plot_tensors(paths, output)
