import math
import matplotlib.pyplot as plt
from pprint import pprint

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torchvision import transforms

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(768, 100)
        )
        
    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out

        
def weights_init(m):
    if hasattr(m, 'weight'):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, 'bias'):
        m.bias.data.uniform_(-0.5, 0.5)

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


device = ('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on %s"%device)


tp = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
])

tt = transforms.ToPILImage()

original_data = Image.open("image/harry_ground_truth.JPG")
original_data = tp(original_data).to(device)

original_data = original_data.view(1, *original_data.size())
original_label = torch.Tensor([0]).long().to(device)
original_label = original_label.view(1, )
original_one_shot_label = F.one_hot(original_label, num_classes=100)

plt.imshow(tt(original_data[0].cpu()))
plt.plot("Ground Truth")
plt.show()


n_net = LeNet().to(device)

torch.manual_seed(1234)

n_net.apply(weights_init)

criterion = cross_entropy_for_onehot

# Original Gradient
pred = n_net(original_data)
y = criterion(pred, original_one_shot_label)
dy_dx = torch.autograd.grad(y, n_net.parameters())

original_dy_dx = list((_.detach().clone() for _ in dy_dx))

# Dummy Label and Data
dummy_data = torch.randn(original_data.size()).to(device).requires_grad_(True)
dummy_label = torch.randn(original_one_shot_label.size()).to(device).requires_grad_(True)

plt.imshow(tt(dummy_data[0].cpu()))
plt.plot("Random Init")
plt.show()

optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

history = []
for i in range(300):
    def closure():
        optimizer.zero_grad()
        
        dummy_pred = n_net(dummy_data) 
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label) 
        dummy_dy_dx = torch.autograd.grad(dummy_loss, n_net.parameters(), create_graph=True)
        
        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, original_dy_dx): 
            grad_diff += ((gx - gy) ** 2).sum()
        grad_diff.backward()
        
        return grad_diff
    
    optimizer.step(closure)
    if i % 10 == 0: 
        current_loss = closure()
        print(i, "%.4f" % current_loss.item())
        history.append(tt(dummy_data[0].cpu()))

plt.figure(figsize=(12, 8))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    plt.imshow(history[i])
    plt.title("iter=%d" % (i * 10))
    plt.axis('off')

plt.show()
        