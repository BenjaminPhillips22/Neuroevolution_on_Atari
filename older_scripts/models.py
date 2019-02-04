import torch
import torch.nn as nn
# from torch.autograd import Variable
import torch.nn.functional as F


class SmallModel(nn.Module):
    """
    A simple model....
    """
    def __init__(self, seed):
        super().__init__()

        self.seed = seed
        torch.manual_seed(seed)

        self.conv1 = nn.Conv2d(4, 16, (8, 8), 4)
        self.conv2 = nn.Conv2d(16, 32, (4, 4), 2)
        self.dense = nn.Linear(1152, 64)
        self.out = nn.Linear(64, 18)

        self.add_tensors = {}
        for name, tensor in self.named_parameters():
            if tensor.size() not in self.add_tensors:
                self.add_tensors[tensor.size()] = torch.Tensor(tensor.size())
            if 'weight' in name:
                nn.init.kaiming_normal_(tensor)
            else:
                tensor.data.zero_()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(1, -1)
        x = F.relu(self.dense(x))
        return self.out(x)

