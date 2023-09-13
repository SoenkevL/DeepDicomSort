import torch.nn as nn
import torch.nn.functional as F
import monai

trainTransforms = monai.transforms.Compose(
    [
        monai.transforms.LoadImaged(keys=['image'],image_only=True,reader='NibabelReader'),
        monai.transforms.EnsureTyped(keys=['image']),
        monai.transforms.EnsureChannelFirstd(keys=['image'])
    ]
)


class Net(nn.Module): #first version ignores the additional 4d parameter but should be easy to add later
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=3,padding=0)
        self.relu = nn.PReLU(num_parameters=1,init=0.25) #check with paper
        self.batchnorm32 = nn.BatchNorm2d(num_features=32)
        self.batchnorm64 = nn.BatchNorm2d(num_features=64)
        self.batchnorm128 = nn.BatchNorm2d(num_features=128)
        self.droput = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(in_features=1600, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=5) #this needs to be adapted to the target classes
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm32(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm32(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.batchnorm64(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.batchnorm64(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv5(x)
        x = self.batchnorm64(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.batchnorm64(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.droput(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.droput(x)
        x = self.fc2(x)
        output = F.softmax(x,dim=0)
        return output


def simpleTrain(model, loss_function, train_dataloader, val_dataloader, optimizer, epochs, device='cpu', val_freq=1):
    train_loss = []
    val_loss = []

    for epoch in range(epochs):
        model.train()
        steps = 0
        epoch_loss = 0

        for batch in train_dataloader:
            optimizer.zero_grad()
            images = batch['image'].float().to(device)
            labels = batch['label'].float().to(device)
            output = model(images)
            # print(output)
            # print(labels)
            loss = loss_function(output,labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            steps += 1

        train_loss.append(epoch_loss/steps)

        # validation loop
        if epoch % val_freq == 0:
            steps = 0
            val_epoch_loss = 0
            model.eval()
            for batch in val_dataloader:
                images = batch['image'].float().to(device)
                labels = batch['label'].float().to(device)
                output = model(images)
                loss = loss_function(output, labels)
                val_epoch_loss += loss.item()
                steps += 1
            val_loss.append(val_epoch_loss/steps)
    print(f'finished training successfully with final validation loss of {val_loss[-1]}')
    return train_loss, val_loss, model