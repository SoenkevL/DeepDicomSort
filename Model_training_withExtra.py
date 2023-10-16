import monai
import os
import torch
import yaml
import datetime
import wandb
import numpy as np
from Pytorch_monai.secrets import wandbkey
import Pytorch_monai.Model_and_transforms as MF
import Pytorch_monai.Utils as Utils
import json
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def prepareModelAndCallbacks(N_train_classes,device='cpu',initWeights=None):
    if initWeights:
        model = Utils.updateModelDictForTransferLearning(initWeights,MF.Net(n_outputclasses=N_train_classes)).to(device=device)
    else:
        model = MF.Net_withExtra(n_outputclasses=N_train_classes).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9,0.999),eps=1e-7,amsgrad=False)
    loss_function = torch.nn.CrossEntropyLoss()
    rop = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,factor=0.1,patience=3,min_lr=1e-6,verbose=1)
    return model, optimizer, loss_function, rop


def prepareData(train_label_file, batch_size,N_train_classes, crv, crt):
    # load imagefilenames and onehot encoded labels
    train_image_IDs, train_image_labels, N_train_classes, extra_inputs = Utils.load_labels(train_label_file,nb_classes=N_train_classes)
    print("Detected %d classes in training data" % N_train_classes)

    #initialize monai transforms
    trainTransforms = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=['image'],image_only=True,reader='NibabelReader'),
            monai.transforms.EnsureTyped(keys=['image']),
            monai.transforms.EnsureChannelFirstd(keys=['image'])
        ]
    )
    valTransforms = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=['image'],image_only=True,reader='NibabelReader'),
            monai.transforms.EnsureTyped(keys=['image']),
            monai.transforms.EnsureChannelFirstd(keys=['image'])
        ]
    )

    #create data dicitionaries
    train_image_labels_noh = np.argmax(train_image_labels,axis=1)
    train_data_dict = [{"image":image_name,"label":label, 'extra':extra} for image_name, label, extra in zip(train_image_IDs,train_image_labels, extra_inputs)]
    train_data_dict, val_data_dict = train_test_split(train_data_dict,stratify=train_image_labels_noh,shuffle=True,random_state=42,test_size=0.1)

    # do some checks on the label distribution
    trainLabelList = [np.argmax(dictItem['label']) for dictItem in train_data_dict]
    print(np.unique(trainLabelList,return_counts=True))
    valLabelList = [np.argmax(dictItem['label']) for dictItem in val_data_dict]
    print(np.unique(valLabelList,return_counts=True))

    #create datasets and loaders
    train_ds = monai.data.CacheDataset(data=train_data_dict,transform=trainTransforms,cache_rate=crt,num_workers=4,progress=True)
    val_ds = monai.data.CacheDataset(data=val_data_dict,transform=valTransforms,cache_rate=crv,num_workers=4,progress=True)
    train_loader = monai.data.DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=0)
    val_loader = monai.data.DataLoader(val_ds,batch_size=batch_size,shuffle=True,num_workers=0)
    return train_loader, val_loader, trainTransforms, valTransforms


#define training loop
def train(model, loss_function, train_dataloader, val_dataloader, optimizer, rop, epochs, model_name, output_folder, device='cpu', val_freq=1):
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
            extraInputs = batch['extra'].float().to(device)
            output = model(images, extraInputs)
            loss = loss_function(output,labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            steps += 1

        train_loss.append(epoch_loss/steps)
        rop.step(train_loss[-1]) #learning rate scheduler
        #implement early stopping
        if len(train_loss) > 6:
            train_loss_diff = np.abs(train_loss[-6])-np.abs(np.min(train_loss[-5:]))
            if train_loss_diff < 0.0001:
                print(f'finished training early with final training loss of {train_loss[-1]} and a loss difference of {train_loss_diff}')
                return train_loss, val_loss, bestModel
        #end of early stopping

        # validation loop
        if epoch % val_freq == 0:
            steps = 0
            val_epoch_loss = 0
            model.eval()
            for batch in val_dataloader:
                images = batch['image'].float().to(device)
                labels = batch['label'].float().to(device)
                extraInputs = batch['extra'].float().to(device)
                output = model(images, extraInputs)
                loss = loss_function(output, labels)
                val_epoch_loss += loss.item()
                steps += 1
            val_loss.append(val_epoch_loss/steps)
            #implement best model saving
            if val_loss[-1] < best_val_loss:
                best_val_loss = val_loss[-1]
                bestModel = model
                torch.save(model,os.path.join(output_folder,model_name+'.pt'))  
        #log the training process
        Utils.log_to_wandb(epoch,train_loss[-1],val_loss[-1])

    print(f'finished training successfully with final validation loss of {val_loss[-1]}')
    return train_loss, val_loss, bestModel



def main(configFile='config.yaml'):
    ### intialize from config file
    with open(configFile, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    train_label_file = cfg['training']['train_label_file']
    train_labelmap_file = cfg['model']['label_map_file']
    output_folder = cfg['training']['output_folder']
    batch_size = cfg['network']['batch_size']
    nb_epoch = cfg['network']['nb_epoch']
    transfer_weights = cfg['training']['transfer_weights_path']
    cache_rate_train = cfg['training']['cache_rate_train']
    cache_rate_val = cfg['training']['cache_rate_val']

    ##label map
    with open(train_labelmap_file) as labelmap:
        label_map = json.load(labelmap)
    N_train_classes = len(label_map.keys())

    ## setup gpu and model name
    gpu = Utils.chooseDevice(verbose=True)
    print(gpu)
    now = str(datetime.datetime.now()).replace(' ', '_')
    model_name = 'DDS_model_epochs' + str(nb_epoch) + '_time_' + now

    #initialize model and data
    model, optimizer, loss_function, rop = prepareModelAndCallbacks(N_train_classes,gpu,transfer_weights)
    train_loader, val_loader, trainTransforms, valTransforms = prepareData(train_label_file=train_label_file,batch_size=batch_size,N_train_classes=N_train_classes, crv=cache_rate_val, crt=cache_rate_train)

    ## setup logging to wandb
    wandb.login(key=wandbkey)
    run = wandb.init(
        project='pytorch_DDS',
        name='training_'+model_name,
        config={
            'train_label_file':train_label_file,
            'label_map':label_map,
            'loss function': str(loss_function),
            'optimizer': str(optimizer),
            'train_transform': Utils.from_compose_to_list(trainTransforms),
            'val_transform': Utils.from_compose_to_list(valTransforms),
            'train_batch_size': train_loader.batch_size,
            'val_batch_size': val_loader.batch_size,
        }
    )
    run_id = run.id # We remember here the run ID to be able to write the evaluation metrics

    trainloss, valloss, model = train(model,loss_function,train_loader,val_loader,optimizer,rop,nb_epoch,device=gpu,val_freq=1, model_name=model_name, output_folder=output_folder)
    



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='This is Model training for the specified config parameters')
    parser.add_argument('-c','--configFile', action='store',metavar='c', help='pass here the config file path (from root or absolute) that should be used with your program')
    args = parser.parse_args()
    configFile = args.configFile
    main(configFile)
    print('finished model training')



