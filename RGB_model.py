import torch
import torch.nn as nn
import torch.nn.functional as F

'''
PointNet AutoEncoder
Learning Representations and Generative Models For 3D Point Clouds
https://arxiv.org/abs/1707.02392
'''

class PointCloudAE(nn.Module):
    def __init__(self, point_size, latent_size):
        super(PointCloudAE, self).__init__()
        
        self.latent_size = latent_size
        self.point_size = point_size
        
        self.conv1 = torch.nn.Conv1d(6, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.latent_size, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.latent_size)
        
        self.dec1 = nn.Linear(self.latent_size,256)
        self.dec2 = nn.Linear(256,256)
        #change output to 6
        self.dec3 = nn.Linear(256,self.point_size * 6)

    def encoder(self, x): 
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.latent_size)
        return x
    
    def decoder(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)
        x = x.view(-1, self.point_size, 6)
        # Avoid in-place modification by using a new tensor for clamped values
        clamped_rgb = torch.clamp(x[..., 3:], 0.0, 1.0)
        x = torch.cat((x[..., :3], clamped_rgb), dim=-1)  # Combine spatial and clamped RGB values
        return x
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

"""

My notes to self:

I havennt changed the network's size or latent vector's
size for now even though the "amount of information"
is esstinally doubled, may  have to depending on 
results

"""