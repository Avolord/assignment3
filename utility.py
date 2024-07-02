import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# import plotting utilities
import matplotlib.pyplot as plt
from IPython import display
# define constants
import math # needed for math.pi
eps = 0.00001
device = "cuda"



def display_io(axes, preds, labels, losses, epoch):
    """
    Displays the input polygon, predicted and ground truth images, and training loss after each epoch.
    
    Args:
    axes (list): List of Matplotlib axes objects for plotting.
    preds (dict): Dictionary containing predicted tensors. Keys can be 'img' and/or 'polygon'.
    labels (dict): Dictionary containing ground truth tensors. Keys are 'img' and 'polygon'.
    losses (list): List of training loss values.
    epoch (int): Current epoch number.
    
    Returns:
    None
    """


    # render the first image in the batch after each epoch
    for ax in axes:
        ax.cla()
    bi = 0 #epoch % batch_size
    axi = 0

    points_gt = labels['polygon'][bi].cpu()
    axes[axi].fill(*points_gt, edgecolor='k', fill=True) # this command closes the loop
    axes[axi].plot(*points_gt.cpu()[:,0],'.',ms=10,color='red') # mark the first vertext to identify issue
    #axes[axi].plot(*points_gt.cpu(),'.') # this command closes the loop
    axes[axi].set_title('Input polygon')
    axi +=1

    if 'img' in preds:
        p_img_pil = torchvision.transforms.ToPILImage()(preds['img'][bi].cpu())
        axes[axi].imshow(p_img_pil)
        axes[axi].set_title('Rendered image')
        axi +=1

        l_img_pil = torchvision.transforms.ToPILImage()(labels['img'][bi].cpu())
        axes[axi].imshow(l_img_pil)
        axes[axi].set_title('Ground truth image')
        axi +=1

    if 'polygon' in preds:
        axes[axi].plot(*preds['polygon'][bi].detach().cpu()[:,0],'.',ms=10,color='red') # mark the first vertext to identify issue
        axes[axi].fill(*preds['polygon'][bi].detach().cpu(), edgecolor='k', fill=True) # this command closes the loop
        axes[axi].set_title('Output polygon')
        axi += 1

        axes[axi].fill(*points_gt.cpu(), edgecolor='gray', fill=False) # this command closes the loop
        axes[axi].plot(*preds['polygon'][bi].detach().cpu(),".")
        axes[axi].set_title('Output pointcloud (GT in gray)')
        axi += 1

    axes[axi].plot(losses)
    axes[axi].set_yscale('log')
    axes[axi].set_title('Training loss')
    axes[axi].set_xlabel("Gradient iterations")

    display.clear_output(wait=True)
    display.display(plt.gcf())
    
    #save the plot to a file in ./results
    plt.savefig('results/plot.png'.format(epoch))
    
    print("Plot after epoch {} (iteration {})".format(epoch, len(losses)))


def dict_to_device(dict_in, device):
    """
    Moves all tensors in a dictionary to a specified device (e.g., GPU or CPU).

    Args:
    dict_in (dict): Dictionary containing tensors as values.
    device (torch.device or str): The device to which the tensors should be moved.

    Returns:
    dict: A new dictionary with all tensors moved to the specified device.
    """
    return {k:v.to(device) for k,v in dict_in.items()}



def roll_dim2(x, n=1):
    """
    Rolls the elements along the third dimension of a 3D tensor.
    
    Args:
    x (torch.Tensor): Input tensor with at least three dimensions.
    n (int): Number of positions to roll. Default is 1.
    
    Returns:
    torch.Tensor: Tensor with elements rolled along the third dimension.
    """
    return torch.cat((x[:, :, -n:], x[:, :, :-n]), dim=2)

def augment_polygon(poly):
    """
    Randomly shifts the starting point of a polygon by rolling its coordinates.
    
    Args:
    poly (torch.Tensor): Tensor representing the polygon coordinates with shape (2, N), 
                         where N is the number of points.
    
    Returns:
    torch.Tensor: Tensor representing the augmented polygon coordinates.
    """
    # Get the number of points in the polygon
    num_points = poly.shape[-1]
    
    # Generate a random number between 0 and the number of points - 1
    random_number = torch.LongTensor(1).random_(0, num_points).item()
    
    # Roll the polygon coordinates by the random number of positions
    poly = roll_dim2(poly, n=random_number)
    
    return poly


# main training loop, will be used throughout this assignment
def train(train_loader, network, loss_fn, num_training_epochs, key="img", augmentation=False, display_results=True):
    losses = []
    
    fig=plt.figure(figsize=(20, 5), dpi= 80, facecolor='w', edgecolor='k')
    axes=fig.subplots(1,4)
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    for epoch in range(num_training_epochs):
        iterator = iter(train_loader)
        network.train()
        for i in range(len(train_loader)):
            batch_cpu = next(iterator)
            batch_size = batch_cpu[key].shape[0]
            if augmentation:
                batch_cpu['polygon'] = augment_polygon(batch_cpu['polygon'])

            batch = dict_to_device(batch_cpu, device)

            preds = network(batch)
            
            loss = loss_fn(preds[key], batch[key])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if epoch % max(num_training_epochs//1000,1) == 0:
            if display_results:
                display_io(axes, preds, batch, losses, epoch)
    display.clear_output(wait=True)


############################### Utility_Section for part II ###########################################

# two-sided loss on the distance of every point to its nearest neighbor
def chamfer_distance(pred, label):
    batch_size, _, num_points = pred.shape

    # Expand dimensions to compute pairwise distances
    pred_exp = pred.unsqueeze(2).expand(batch_size, 2, num_points, num_points)
    label_exp = label.unsqueeze(3).expand(batch_size, 2, num_points, num_points)

    # Compute pairwise distances
    distances = torch.norm(pred_exp - label_exp, dim=1)  # Shape: (batch_size, num_points, num_points)

    # Find nearest neighbors
    min_pred_to_label, _ = distances.min(dim=2)  # Shape: (batch_size, num_points)
    min_label_to_pred, _ = distances.min(dim=1)  # Shape: (batch_size, num_points)

    # Compute Chamfer distance
    chamfer_dist = min_pred_to_label.mean(dim=1) + min_label_to_pred.mean(dim=1)
    return chamfer_dist.mean()  # Return mean Chamfer distance over the batch



############################### Utility_Section for part III ###########################################

# Task-III: Beyond Chamfer 
def start_invariant_MSE(pred, label):
    min_loss = None
    
    # (Done) TODO TASK III
    # Hint1: you can use the following helper function that shifts/rolls a tensor along the third dimension
    def roll_2(x, n=1):
      return torch.cat((x[:,:,-n:], x[:,:,:-n]),dim=2)

    # Hint2: one for loop should be sufficient. Try to use tensor operations as much as possible
    batch_size, _, num_points = pred.shape
    min_loss = None

    # Calculate MSE for all possible starting positions
    for shift in range(num_points):
        rolled_pred = roll_2(pred, shift)
        mse_loss = torch.mean((rolled_pred - label) ** 2, dim=[1, 2])  # Shape: (batch_size,)
        if min_loss is None:
            min_loss = mse_loss
        else:
            min_loss = torch.min(min_loss, mse_loss)

    return min_loss.mean()  # Return mean MSE over the batch
    
    return min_loss

def interpolate_points(M1, M2, lambdas):
    return [(1 - lam) * M1 + lam * M2 for lam in lambdas]

def interpolate_latent(h1, h2, lambdas):
    return [(1 - lam) * h1 + lam * h2 for lam in lambdas]

if __name__ == "__main__":
    #Task III_b
    # Test with toy examples
    pred = torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.float32)  # Shape: (1, 2, 3)
    label = torch.tensor([[[2, 3, 1], [5, 6, 4]]], dtype=torch.float32)  # Shape: (1, 2, 3)

    loss = start_invariant_MSE(pred, label)
    print(f'Start-invariant MSE: {loss.item()}')

    # Test with batches of polygons
    batch_pred = torch.tensor([
        [[1, 2, 3], [4, 5, 6]],  # Polygon 1
        [[7, 8, 9], [10, 11, 12]]  # Polygon 2
    ], dtype=torch.float32)  # Shape: (2, 2, 3)

    batch_label = torch.tensor([
        [[2, 3, 1], [5, 6, 4]],  # Polygon 1
        [[9, 7, 8], [12, 10, 11]]  # Polygon 2
    ], dtype=torch.float32)  # Shape: (2, 2, 3)

    batch_loss = start_invariant_MSE(batch_pred, batch_label)
    print(f'Start-invariant MSE for batch: {batch_loss.item()}')
    