import torch 
import torch.nn as nn 

# an improved autoencoder that uses *no* fully-connected layer
class PolygonAE(nn.Module):
    def __init__(self, num_points, bottleneck_width):
        super(PolygonAE, self).__init__()
        
        # (Done) TODO, TASK IV: Avoid any fully-connected layer in the encoder
        self.encoder = nn.Sequential(
            # (Done) TODO create layers in here or after the sequential block

            # Hint: you can use padding_mode='circular' in convolutions
            nn.Conv1d(2, 64, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.MaxPool1d(2),

            nn.Conv1d(256, 512, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.MaxPool1d(2),

            nn.Conv1d(512, bottleneck_width, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm1d(bottleneck_width),
            nn.ReLU(True),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling to get bottleneck features
        )

        channels_decoder = 128

        # It is OK to maintain the following decoder
        self.fc2a = nn.Linear(bottleneck_width, channels_decoder)
        self.fc2c = nn.Linear(channels_decoder, 2*num_points)

    def encode(self, dictionary):
        x_NCW = dictionary['polygon']
        return self.encoder(x_NCW)

    def decode(self, z):
        batch_size = z.shape[0]
        z_flat = z.view(batch_size, -1)
        h2 = nn.ReLU()(self.fc2a(z_flat))
        h2 = self.fc2c(h2)    
        y_NCW = h2.view([batch_size,2,-1])
        return {'polygon': y_NCW}

    def forward(self, dictionary):
        z = self.encode(dictionary)        
        out_dict = self.decode(z)
        return out_dict
