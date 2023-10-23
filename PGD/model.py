import torch
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self,batch_size):
        super(Net, self).__init__()
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  #32*30*30
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  #64*13*13
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)   #64*4*4
        self.conv2_drop = nn.Dropout2d()
        self.dense1 = torch.nn.Linear(16*64, 1024)
        self.dense2 = torch.nn.Linear(1024, 64)
        self.dense3 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.relu(self.conv3(x))
        x = x.reshape(-1,1024)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

