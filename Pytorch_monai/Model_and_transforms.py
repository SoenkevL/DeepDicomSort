import torch.nn as nn
import torch.nn.functional as F
import torch
class Net(nn.Module): #first version ignores the additional 4d parameter but should be easy to add later
    def __init__(self, n_outputclasses=6):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.maxpool12 = nn.MaxPool2d(kernel_size=3,stride=3,padding=0)
        self.maxpool34 = nn.MaxPool2d(kernel_size=3,stride=3,padding=0)
        self.maxpool56 = nn.MaxPool2d(kernel_size=3,stride=3,padding=0)
        self.relu1 = nn.PReLU(num_parameters=32,init=0.25) #check with paper
        self.relu2 = nn.PReLU(num_parameters=32,init=0.25)
        self.relu3 = nn.PReLU(num_parameters=64,init=0.25)
        self.relu4 = nn.PReLU(num_parameters=64,init=0.25)
        self.relu5 = nn.PReLU(num_parameters=64,init=0.25)
        self.relu6 = nn.PReLU(num_parameters=64,init=0.25)
        self.batchnorm32_1 = nn.BatchNorm2d(num_features=32)
        self.batchnorm32_2 = nn.BatchNorm2d(num_features=32)
        self.batchnorm64_3 = nn.BatchNorm2d(num_features=64)
        self.batchnorm64_4 = nn.BatchNorm2d(num_features=64)
        self.batchnorm64_5 = nn.BatchNorm2d(num_features=64)
        self.batchnorm64_6 = nn.BatchNorm2d(num_features=64)
        self.droput1 = nn.Dropout(p=0.4)
        self.droput2 = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(in_features=1601, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=n_outputclasses) #this needs to be adapted to the target classes
        self.flatten = nn.Flatten()

    def forward(self, x, extra):
        x = self.conv1(x)
        x = self.batchnorm32_1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batchnorm32_2(x)
        x = self.relu2(x)
        x = self.maxpool12(x)
        x = self.conv3(x)
        x = self.batchnorm64_3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.batchnorm64_4(x)
        x = self.relu4(x)
        x = self.maxpool34(x)
        x = self.conv5(x)
        x = self.batchnorm64_5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.batchnorm64_6(x)
        x = self.relu6(x)
        x = self.maxpool56(x)
        x = self.flatten(x)
        x = self.droput1(torch.cat((x,extra),1)) #possibly need to change the dimension along which convatenates
        x = self.fc1(x)
        x = F.relu(x)
        x = self.droput2(x)
        x = self.fc2(x)
        output = F.softmax(x,dim=1)
        return output


