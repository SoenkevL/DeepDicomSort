import monai
import os
import torch
import numpy as np
import yaml
import datetime
import wandb
from Pytorch_monai.secrets import wandbkey
import Pytorch_monai.Model_and_transforms as MF
import Pytorch_monai.Utils as Utils
import json
import logging
from tqdm import tqdm

### intialize from config file
with open('./config.yaml', 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

train_label_file = cfg['training']['train_label_file']
train_labelmap_file = cfg['training']['label_map_file']
x_image_size = cfg['data_preparation']['image_size_x']
y_image_size = cfg['data_preparation']['image_size_y']
output_folder = cfg['training']['output_folder']
batch_size = cfg['network']['batch_size']
nb_epoch = cfg['network']['nb_epoch']

wandb.login(key=wandbkey)

## setup gpu and model name
gpu = Utils.chooseDevice(verbose=True)
print(gpu)
now = str(datetime.datetime.now()).replace(' ', '_')
model_name = 'DDS_model_epochs' + str(nb_epoch) + '_time_' + now

##label map
with open(train_labelmap_file) as labelmap:
    label_map = json.load(labelmap)

# load imagefilenames and onehot encoded labels
train_image_IDs, train_image_labels, N_train_classes, extra_inputs = Utils.load_labels(train_label_file)
print("Detected %d classes in training data" % N_train_classes)

#initialize monai transforms
trainTransforms = monai.transforms.Compose(
    [
        monai.transforms.LoadImaged(keys=['image'],image_only=True,reader='NibabelReader'),
        monai.transforms.EnsureTyped(keys=['image']),
        monai.transforms.EnsureChannelFirstd(keys=['image'],channel_dim=0)
    ]
)
valTransforms = monai.transforms.Compose(
    [
        monai.transforms.LoadImaged(keys=['image'],image_only=True,reader='NibabelReader'),
        monai.transforms.EnsureTyped(keys=['image']),
        monai.transforms.EnsureChannelFirstd(keys=['image'],channel_dim=0)
    ]
)

#create data dicitionaries and data loader to which monai transforms can be applied (transforms stored in Model_and_transforms.py)
train_data_dict = [{"image":image_name,"label":label} for image_name, label in zip(train_image_IDs,train_image_labels)]
val_data_dict = train_data_dict[-1000:]
train_data_dict = train_data_dict[:-1000] #this can be optimized to shuffle beforehand for example
train_ds = monai.data.CacheDataset(data=train_data_dict,transform=trainTransforms,cache_rate=0.5,num_workers=4,progress=True)
val_ds = monai.data.CacheDataset(data=val_data_dict,transform=valTransforms,cache_rate=1,num_workers=4,progress=True)
train_loader = monai.data.DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=0)
val_loader = monai.data.DataLoader(val_ds,batch_size=batch_size,shuffle=True,num_workers=0)
#intialize model, optimizer, loss
model = MF.Net(n_outputclasses=N_train_classes).to(device=gpu)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9,0.999),eps=1e-7,amsgrad=False)
loss_function = torch.nn.CrossEntropyLoss()

#setup callbacks
rop = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,factor=0.1,patience=3,min_lr=1e-6,verbose=1)

## setup logging to wandb
run = wandb.init(
    project='pytorch_DDS',
    name='training_'+model_name,
    config={
        'label_map':label_map,
        'loss function': str(loss_function),
        'optimizer': str(optimizer),
        'lr': optimizer.param_groups[0]["lr"],
        'train_transform': Utils.from_compose_to_list(trainTransforms),
        'val_transform': Utils.from_compose_to_list(valTransforms),
        'train_batch_size': train_loader.batch_size,
        'val_batch_size': val_loader.batch_size,
    }
)
# Do not hesitate to enrich this list of settings to be able to correctly keep track of your experiments!
# For example you should add information on your model...

run_id = run.id # We remember here the run ID to be able to write the evaluation metrics

#define training loop
def train(model, loss_function, train_dataloader, val_dataloader, optimizer, epochs, device=gpu, val_freq=1):
    print('\n\n------------------start training------------------------------\n\n')
    train_loss = []
    val_loss = []
    best_val_loss = 1000

    for epoch in tqdm(range(epochs)):
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
        rop.step(train_loss[-1]) #learning rate scheduler

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
            #implement best model saving
            if val_loss[-1] < best_val_loss:
                best_val_loss = val_loss[-1]
                bestModel = model
            #implement early stopping
            if len(val_loss) > 6:
                if np.abs(val_loss[-6])-np.abs(np.min(val_loss[-5:-1])) < 0.01:
                    print(f'finished training early with final validation loss of {val_loss[-1]}')
                    return train_loss, val_loss, bestModel
            #end of early stopping
        #log the training process
        Utils.log_to_wandb(epoch,train_loss[-1],val_loss[-1])

    print(f'finished training successfully with final validation loss of {val_loss[-1]}')
    return train_loss, val_loss, bestModel

#train
trainloss, valloss, model = train(model,loss_function,train_loader,val_loader,optimizer,nb_epoch,device=gpu,val_freq=2)

#save model
torch.save(model,os.path.join(output_folder,model_name+'.pt'))





