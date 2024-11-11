import math
import matplotlib.pyplot as plt
from pprint import pprint

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torchvision import transforms

# Taken directly from (just testing): https://github.com/mit-han-lab/dlg/blob/master/models/vision.py
class LetNet(nn.Module):
    def __init__(self):
        super(LetNet, self).__init__()
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
    
# Taken directly from (just testing): https://github.com/mit-han-lab/dlg/blob/master/models/vision.py
def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)
# Taken directly from, (just testing): https://github.com/mit-han-lab/dlg/blob/master/utils.py
def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

# Taken directly from (just testing): https://github.com/mit-han-lab/dlg/blob/master/utils.py
def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


device = ('cuda' if torch.cuda.is_available() else 'cpu')
print("Device running on %s"%device)


tp = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
])
tt = transforms.ToPILImage()

gt_data = Image.open('image/harry_ground_truth.JPG')
gt_data = tp(gt_data).to(device)
gt_data = gt_data.view(1, *gt_data.size())

gt_label = torch.Tensor([0]).long().to(device)
gt_label = gt_label.view(1, )
gt_oneshot_label = label_to_onehot(gt_label)


plt.imshow(tt(gt_data[0].cpu()))
plt.plot("Ground Truth")
plt.show()

n_net = LetNet().to(device)

torch.manual_seed(1066)

n_net.apply(weights_init)
criterion = cross_entropy_for_onehot

pred = n_net(gt_data)
y = criterion(pred, gt_oneshot_label)
dy_dx = torch.autograd.grad(y, n_net.parameters())
original_dy_dx = list((_.detach().clone() for _ in dy_dx))

dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
dummy_label = torch.randn(gt_oneshot_label.size()).to(device).requires_grad_(True)

plt.imshow(tt(dummy_data[0].cpu()))
plt.plot("Random Init")
plt.show()

optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

history = []

for i in range(1000):
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
print(len(history))

num_images = len(history)
num_cols = 10
num_rows = math.ceil(num_images / num_cols)  # Calculate the required number of rows

plt.figure(figsize=(12, num_rows * 2.5))

for i in range(100):
    plt.subplot(num_rows, num_cols, i + 1)
    plt.imshow(history[i])
    plt.title("iter=%d" % (i * 10))
    plt.axis('off')

plt.show()
