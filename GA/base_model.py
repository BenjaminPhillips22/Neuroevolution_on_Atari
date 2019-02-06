# In the GA folder. This contains the BigModel
# which takes 'seed_dict' and creates the model
# using different seeds for different weight layers.

import torch
import torch.nn as nn
import torch.nn.functional as F


class BigModel(nn.Module):
    """
    takes 'seed_dict' and creates the model
    using different seeds for different weight layers.

    """
    def __init__(self, seed_dict):
        super().__init__()

        self.seed_dict = seed_dict

        self.conv1 = nn.Conv2d(4, 32, (8, 8), 4)
        self.conv2 = nn.Conv2d(32, 64, (4, 4), 2)
        self.conv3 = nn.Conv2d(64, 64, (3, 3), 1)
        self.dense = nn.Linear(4*4*64, 512)
        self.out = nn.Linear(512, 18)

        for name, tensor in self.named_parameters():

            tensor.data.zero_()

            if name in self.seed_dict.keys():

                kai = True

                for s in self.seed_dict[name]:

                    torch.manual_seed(s)

                    to_add = torch.Tensor(tensor.size()).data.zero_()
                    to_add.normal_(0.0, 0.005)

                    tensor.data.add_(to_add)

                    if kai:
                        nn.init.kaiming_normal_(tensor)
                        kai = False

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(1, -1)
        x = F.relu(self.dense(x))
        return self.out(x)
