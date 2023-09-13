import pandas as pd
import monai
import os
import torch
import numpy as np
import yaml
import datetime

import Pytorch_monai.Model_and_transforms as MF
import Pytorch_monai.Utils as Utils

### intialize from config file
with open('./config.yaml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

train_label_file = cfg['training']['train_label_file']
x_image_size = cfg['data_preparation']['image_size_x']
y_image_size = cfg['data_preparation']['image_size_y']
output_folder = cfg['training']['output_folder']
batch_size = cfg['network']['batch_size']
nb_epoch = cfg['network']['nb_epoch']

## setup gpu and model name
gpu = Utils.chooseDevice()
now = str(datetime.datetime.now()).replace(' ', '_')
model_name = 'DDS_model_epochs' + str(nb_epoch) + '_time_' + now

# load imagefilenames and onehot encoded labels
train_image_IDs, train_image_labels, N_train_classes, extra_inputs = Utils.load_labels(train_label_file)
print("Detected %d classes in training data" % N_train_classes)

#create data dicitionaries and data loader to which monai transforms can be applied (transforms stored in Model_and_transforms.py)
train_data_dict = [{"image":image_name,"label":label} for image_name, label in zip(train_image_IDs,train_image_labels)]
val_data_dict = train_data_dict[-10:]
train_data_dict = train_data_dict[:-10] #this can be optimized to shuffle beforehand for example
train_ds = monai.data.CacheDataset(data=train_data_dict,transform=MF.trainTransforms,cache_rate=1,num_workers=4)
val_ds = monai.data.CacheDataset(data=val_data_dict,transform=MF.trainTransforms,cache_rate=-1,num_workers=4)
train_loader = monai.data.DataLoader(train_ds,batch_size=10,shuffle=True,num_workers=0)
val_loader = monai.data.DataLoader(val_ds,batch_size=10000,shuffle=True,num_workers=0)

#intialize model, optimizer, loss
model = MF.Net().to(device=gpu)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_function = torch.nn.BCEWithLogitsLoss()


#train
trainloss, valloss, model = MF.simpleTrain(model,loss_function,train_loader,val_loader,optimizer,nb_epoch,device=gpu,val_freq=10)






