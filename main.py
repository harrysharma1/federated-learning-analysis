import matplotlib.pyplot as plt
from pprint import pprint

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torchvision import transforms


class LetNet5(nn.Module):
    def __init__(self, n_classes=100):
        super(self,LetNet5).__init__()
        
        # C1: Convolusional layer 6 filters x (5x5) 3 channels(RGB)
        self.conv_1 = nn.Conv2d(3, 6, 5) 
        
        # S2: Adds a pooling layer
        self.pool_1 = nn.AvgPool2d(2, 2) 
        
        # C3: Convolusional layer 16 filters x (5x5)
        self.conv_2 = nn.Conv2d(6, 16, 5)
        
        # S4: Add another pooling layer
        self.pool_2 = nn.AvgPool2d(2, 2)
        
        # C5: Fully Connected layer
        self.fc_1 = nn.Linear(16*5*5, 120)
        
        # F6: Fully Connected layer with 84 units
        self.fc_2 = nn.Linear(120,84)
        
        # Output
        self.output = nn.Linear(84,n_classes)
        
    def forward(self, x):
        # C1 + S2
        x = F.tanh(self.conv_1(x))
        x = self.pool_1(x)
        
        # C3 + S4
        x = F.tanh(self.conv_2(x))
        x = self.pool_2(x)
        
        # Flatten
        x = x.view(-1, 16*5*5)
        
        # Fully Connected Layer
        x = F.tanh(self.fc_1(x))
        x = F.tanh(self.fc_2(x))
        
        # Output layer
        x = self.fc3
        
        return x
    
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print("Device running on %s"%device)

tp = transforms.ToTensor()
tt = transforms.ToPILImage()

gt_data = Image.open('image/harry_ground_truth.JPG')

gt_data = tp(gt_data).to(device)

gt_data = gt_data.view(1, *gt_data.size())
gt_label = torch.Tensor([0]).long().to(device)
gt_label = gt_label.view(1, )

plt.imshow(tt(gt_data[0].cpu()))
plt.plot("Ground Truth")
plt.show()