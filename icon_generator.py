import torch 
import torch.nn as nn 

# network architecture skeleton
class IconGenerator(nn.Module):
    def __init__(self, num_points, channels=32, out_channels=1):
        super(IconGenerator, self).__init__()

        # maps the input points of size (batch dim) x 2 x N
        # to a feature map (batch dim) x (#channels) x 2 x 2 
        self.MLP = nn.Sequential(
            nn.Linear(in_features=num_points*2, out_features=channels * 2*2),
            nn.ReLU(True),
        )

        # define a sequence of upsampling, batch norm, ReLu, etc. to map 2x2 features to 32 x 32 images
        self.main = nn.Sequential(
            # input size: (batch dim) x (#channels) x 2 x 2

            # (Done) TODO, TASK I: define a sequence of suitable layers. Note, you don't have to use nn.Sequential.
            nn.ConvTranspose2d(channels, channels // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(channels // 2, channels // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(channels // 4, channels // 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels // 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(channels // 8, out_channels, kernel_size=4, stride=2, padding=1),

            nn.Sigmoid(),
            # output size: (batch dim) x (#out_channels=1) x 32 x 32
        )
      

    def forward(self, input_dict):
        poly = input_dict['polygon']
        batch_size = poly.shape[0]
        
        print(poly.shape)
        print(poly.view([batch_size,-1]).shape)
        
        img_init = self.MLP(poly.view([batch_size,-1]))
        img = self.main(img_init.view([batch_size,-1,2,2]))
        return {'img': img}