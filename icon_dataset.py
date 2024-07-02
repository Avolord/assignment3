import torch
import h5py
import os
import urllib

# Define constants
dataset_file_name = "ImagerIcon_subset.hdf5"

# Check if the dataset is already downloaded
if not os.path.exists(dataset_file_name):
    print("Downloading dataset")
    urllib.request.urlretrieve("https://www.cs.ubc.ca/~rhodin/20_CPSC_532R_533R/assignments/"+dataset_file_name, dataset_file_name)
    print("Done downloading")
else:
    print("Dataset already present, nothing to be done")

# Define the IconDataset class
class IconDataset(torch.utils.data.Dataset):
    def __init__(self, data_file):
        super(IconDataset, self).__init__()
        print("Loading dataset to memory, can take some seconds")
        with h5py.File(data_file, 'r') as hf:
            self.polygon = torch.from_numpy(hf['polygon'][...])
            self.imgs  = torch.from_numpy(hf['img'][...])[:,:1,:,:]
        print(".. done loading")
        
    def __len__(self):
        return self.polygon.shape[0]
    
    def __getitem__(self, idx):
        # transpose to bring the point dimension in the first place
        poly = self.polygon[idx].T.clone()
        # negate show icons upright, scale to make networks better behaved
        poly[1,:] *= -1
        sample = {'img': self.imgs[idx].float()/255, # shape 1 x H x W
                  'polygon': poly, # shape 2 x N
                  }
        return sample
