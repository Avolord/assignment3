import torch 
import torch.nn as nn 

# simple (but inefficient) polygon autoencoder using fully-connected layers
class AE(nn.Module):
    def __init__(self, num_points, bottleneck_width):
        super(AE, self).__init__()
        max_channels = 128
        
        self.fc1a = nn.Linear(2*num_points, max_channels)
        self.fc1c = nn.Linear(max_channels, bottleneck_width)
        
        self.fc2a = nn.Linear(bottleneck_width, max_channels)
        self.fc2c = nn.Linear(max_channels, 2*num_points)

    def encode(self, dictionary):
        x = dictionary['polygon']
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        h1 = nn.ReLU()(self.fc1a(x))
        return self.fc1c(h1)
    
    def decode(self, z):
        batch_size = z.shape[0]
        h2 = nn.ReLU()(self.fc2a(z))
        h2 = self.fc2c(h2)
        
        y_NCW = h2.view([batch_size,2,-1])
        return {'polygon': y_NCW}

    def forward(self, dictionary):
        z = self.encode(dictionary)        
        poly_dict = self.decode(z)
        return poly_dict