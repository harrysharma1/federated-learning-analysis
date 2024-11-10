import matplotlib.pyplot as plt
from pprint import pprint

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torchvision import transforms


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

tp = transforms.ToTensor()
tt = transforms.ToPILImage()

gt_data = Image.open('image/harry_ground_truth.JPG')

gt_data = tp(gt_data).to(device)

gt_data = gt_data.view(1, *gt_data.size())
gt_label = torch.Tensor([0]).long().to(device)
gt_label = gt_label.view(1, )
gt_oneshot_label = label_to_onehot(gt_label)


plt.imshow(tt(gt_data[0].cpu()))
# plt.plot("Ground Truth")
# plt.show()

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
plt.plot("Check")
plt.show()

optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
