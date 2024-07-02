import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import math
from utility import chamfer_distance, interpolate_latent, interpolate_points, start_invariant_MSE, train
# Import the IconDataset class
from icon_dataset import IconDataset 
import argparse 
# Import the IconGenerator class
from simple_ae import AE 
from polygon_ae import PolygonAE 
from utility import dict_to_device

# Define constants
eps = 0.00001
device = "cuda"
dataset_file_name = "ImagerIcon_subset.hdf5"
num_points = 10  # Adjust as necessary

# Load and split the dataset
icon = IconDataset(dataset_file_name)
print("Number of examples in dataset: {}".format(len(icon)))

val_ratio = 0.05
val_size = int(len(icon) * val_ratio)
indices_val = list(range(0, val_size))
indices_train = list(range(val_size, len(icon)))

val_set = torch.utils.data.Subset(icon, indices_val)
train_set = torch.utils.data.Subset(icon, indices_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size = 32, shuffle=True, drop_last=True)
# Task II
def task_II():

    net_simple = AE(num_points=96, bottleneck_width=10).cuda()

    #loss_fn = torch.nn.MSELoss()
    loss_fn = chamfer_distance #(utility.py) enable this to train with the Chamfer distance

    train(train_loader, net_simple, loss_fn, num_training_epochs=200, key='polygon')


# Task III-b: Debugging 
# train net_simple2, with the start_invariant_MSE-(check utility.py) once you are sure it does the right thing. Compare results visually to the MSE training
def task_III():
    net_simple2 = AE(num_points=96, bottleneck_width=10).cuda()

    train(train_loader, net_simple2, start_invariant_MSE, num_training_epochs=200, key='polygon')

#Task IV
# train a new network to be able to compare results to the initial training
def task_IV():
    #net_graph = AE(num_points=96, bottleneck_width=10).cuda() # (Done) TODO: try this one first
    net_graph = PolygonAE(num_points=96, bottleneck_width=10).cuda() # (Done) TODO: uncomment this to replace the simlpe AE with your dedicated one
    train(train_loader, net_graph, start_invariant_MSE, num_training_epochs=2000, key='polygon', augmentation=True, display_results=False)
    
    # Save the trained model
    torch.save(net_graph.state_dict(), 'trained_model_iv.pth')

def task_V():
    net = PolygonAE(num_points=96, bottleneck_width=10).cuda()
    net.eval()
    
    # Load trained model
    # Assuming the model has been trained and saved
    checkpoint = torch.load('trained_model_iv.pth')
    net.load_state_dict(checkpoint)

    data_iterator = iter(train_loader)
    batch_cpu = next(data_iterator)
    batch = dict_to_device(batch_cpu, device)
    
    #select two random polygons
    i1 = np.random.randint(0, batch['polygon'].shape[0])
    i2 = np.random.randint(0, batch['polygon'].shape[0])
    
    #make sure they are different
    while i1 == i2:
        i2 = np.random.randint(0, batch['polygon'].shape[0])
    
    M1 = batch['polygon'][i1].unsqueeze(0)
    M2 = batch['polygon'][i2].unsqueeze(0)

    lambdas = np.linspace(0, 1, 6)

    # Interpolation in input space
    interpolated_points = interpolate_points(M1, M2, lambdas)

    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    for i, lam in enumerate(lambdas):
        interpolated_point = interpolated_points[i].squeeze().cpu().detach().numpy()
        axes[0, i].plot(interpolated_point[0, :], interpolated_point[1, :], marker='o')
        axes[0, i].set_title(f'λ={lam:.1f}')
        axes[0, i].set_xlim([-1, 1])
        axes[0, i].set_ylim([-1, 1])
        axes[0, i].set_aspect('equal')
        
        # zoom out a bit
        axes[0, i].set_xlim([-2, 2])
        axes[0, i].set_ylim([-2, 2])

    # Interpolation in latent space
    h1 = net.encode({'polygon': M1})  # Encoding M1
    h2 = net.encode({'polygon': M2})  # Encoding M2

    interpolated_latents = interpolate_latent(h1, h2, lambdas)
    for i, lam in enumerate(lambdas):
        interpolated_latent = interpolated_latents[i]
        reconstructed_point = net.decode(interpolated_latent)['polygon'].squeeze().cpu().detach().numpy()
        axes[1, i].plot(reconstructed_point[0, :], reconstructed_point[1, :], marker='o')
        axes[1, i].set_title(f'λ={lam:.1f}')
        axes[1, i].set_xlim([-1, 1])
        axes[1, i].set_ylim([-1, 1])
        axes[1, i].set_aspect('equal')
        
        axes[1, i].set_xlim([-2, 2])
        axes[1, i].set_ylim([-2, 2])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run specific tasks for the model.")
    parser.add_argument('--task', type=str, choices=['ii', 'iii', 'iv','v'], required=True, 
                        help="Specify which task to run: 'ii', 'iii','iv','v'.")

    args = parser.parse_args()
    
    if args.task == 'ii':
        task_II()
    elif args.task == 'iii':
        task_III()
    elif args.task == 'iv':
        task_IV()
    elif args.task == 'v':
        task_V()