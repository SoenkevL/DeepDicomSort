import monai
from torch.utils.data import WeightedRandomSampler
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
import random
import matplotlib.pyplot as plt

def choosetransform(augment=False, slice_scaling=False, device='cpu'):
    if slice_scaling:
        if augment:
            trainTransforms = monai.transforms.Compose(
                [
                    monai.transforms.LoadImaged(keys=['image'],image_only=True,reader='NibabelReader'),
                    monai.transforms.EnsureTyped(keys=['image']),
                    monai.transforms.EnsureChannelFirstd(keys=['image']),
                    monai.transforms.ScaleIntensityd(keys=['image']),
                    monai.transforms.RandAffined(
                        keys=['image'],
                        prob=0.5,
                        rotate_range=0.2,
                        shear_range=((-0.2, 0.2), (-0.2, 0.2)),
                        translate_range=((-20, 20), (-20, 20)),
                        device=device,
                        cache_grid=True
                    ),
                    monai.transforms.RandZoomd(keys=['image'],prob=0.25, min_zoom=0.8, max_zoom=1.2)
                ]
            )
        else:
            trainTransforms = monai.transforms.Compose(
                [
                    monai.transforms.LoadImaged(keys=['image'],image_only=True,reader='NibabelReader'),
                    monai.transforms.EnsureTyped(keys=['image']),
                    monai.transforms.EnsureChannelFirstd(keys=['image']),
                    monai.transforms.ScaleIntensityd(keys=['image'])
                ]
            )
        valTransforms = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image'],image_only=True,reader='NibabelReader'),
                monai.transforms.EnsureTyped(keys=['image']),
                monai.transforms.EnsureChannelFirstd(keys=['image']),
                monai.transforms.ScaleIntensityd(keys=['image']),
            ]
        )
    else:
        if augment:
            trainTransforms = monai.transforms.Compose(
                [
                    monai.transforms.LoadImaged(keys=['image'],image_only=True,reader='NibabelReader'),
                    monai.transforms.EnsureTyped(keys=['image']),
                    monai.transforms.EnsureChannelFirstd(keys=['image']),
                    monai.transforms.RandAffined(
                        keys=['image'],
                        prob=0.5,
                        rotate_range=0.2,
                        shear_range=((-0.2, 0.2), (-0.2, 0.2)),
                        translate_range=((-20, 20), (-20, 20)),
                        device=device,
                        cache_grid=True
                    ),
                    monai.transforms.RandZoomd(keys=['image'],prob=0.25, min_zoom=0.8, max_zoom=1.2)
                ]
            )
        else:
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
    return trainTransforms, valTransforms

def freezeConvLayers(model, verbose=False):
    '''
    freezes all layers which contain 'conv' in their name
    '''
    for name, para in model.named_parameters():
        if 'conv' in name:
            if verbose:
                print(f'freezing layer: {name}')
            para.requires_grad = False
        else:
            continue
    return model

def ensureUnfrozenLayers(model, verbose=False):
    '''
    makes sure all layers of the model are unfroozen
    '''
    for name, para in model.named_parameters():
        if para.requires_grad:
            continue
        else:
            if verbose:
                print(f'unfroze {name}')
            para.requires_grad = True
    return model

def prepareModelAndCallbacks(N_train_classes,device='cpu',initWeights=None, freeze_conv=False, sliceScaling=False):
    '''
    Loads the model with transfer weights if they are given and puts it into the specified device
    Also initializes and returns the optimier, loss function and lr_scheduler
    N_train_classes: How many output classes should the model have
    device: device to run the training on
    initWeights: state dict or model in pytorch standard to load weights for transfer learning
    freeez_conv: if the concolution layers of the transfer weights should be frozen
    '''
    if initWeights:
        model = Utils.updateModelDictForTransferLearning(
            initWeights, MF.Net(n_outputclasses=N_train_classes)
        ).to(device=device)
        if freeze_conv:
            model = freezeConvLayers(model, verbose=True)
            non_frozen_parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(non_frozen_parameters, lr=0.001, betas=(0.9,0.999),eps=1e-7,amsgrad=False)
        else:
            model = ensureUnfrozenLayers(model, True)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9,0.999),eps=1e-7,amsgrad=False)
    else:
        model = MF.Net(n_outputclasses=N_train_classes).to(device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9,0.999),eps=1e-7,amsgrad=False)
    loss_function = torch.nn.CrossEntropyLoss()
    rop = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,factor=0.1,patience=3,min_lr=1e-6,verbose=1)
    return model, optimizer, loss_function, rop




def prepareData(
        train_label_file, batch_size,N_train_classes, crv, crt,
        augment=False, randomWeightedSampling=False, per_slice_normalization=False ,device='cpu'):
    # load imagefilenames and onehot encoded labels
    train_image_IDs, train_image_labels, N_train_classes, extra_inputs = Utils.load_labels(train_label_file,nb_classes=N_train_classes)
    print("Detected %d classes in training data" % N_train_classes)

    #initialize monai transforms
    trainTransforms, valTransforms = choosetransform(augment, per_slice_normalization, device)

    #create data dicitionaries
    train_image_labels_noh = np.argmax(train_image_labels,axis=1)
    train_data_dict = [{"image":image_name,"label":label} for image_name, label in zip(train_image_IDs,train_image_labels)]
    train_data_dict, val_data_dict = train_test_split(train_data_dict,stratify=train_image_labels_noh,shuffle=True,random_state=42,test_size=0.1)

    # do some checks on the label distribution
    trainLabelList = [np.argmax(dictItem['label']) for dictItem in train_data_dict]
    print(np.unique(trainLabelList,return_counts=True))
    valLabelList = [np.argmax(dictItem['label']) for dictItem in val_data_dict]
    print(np.unique(valLabelList,return_counts=True))

    #create datasets and loaders
    train_ds = monai.data.CacheDataset(data=train_data_dict,transform=trainTransforms,cache_rate=crt, cache_num=40000,
                                       num_workers=4,progress=True)
    val_ds = monai.data.CacheDataset(data=val_data_dict,transform=valTransforms,cache_rate=crv,cache_num=40000,
                                     num_workers=4,progress=True)
    if randomWeightedSampling:
        classCounts = [dictItem['label'] for dictItem in train_data_dict]
        classCounts = np.sum(classCounts, axis=0)
        print(f'classCount: {classCounts}')
        weights = 1./classCounts
        weights = [0 if np.isinf(x) else x for x in weights]
        LabelWeights = [weights[x] for x in trainLabelList]
        print(f'Labelweights: {weights}')
        dwrs = WeightedRandomSampler(LabelWeights, len(LabelWeights), replacement=True)
        train_loader = monai.data.DataLoader(train_ds,batch_size=batch_size,num_workers=0,sampler=dwrs)
    else:
        train_loader = monai.data.DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=0)
    val_loader = monai.data.DataLoader(val_ds,batch_size=batch_size,shuffle=True,num_workers=0)
    return train_loader, val_loader, trainTransforms, valTransforms


#define training loop
def train(
        model, loss_function, train_dataloader, val_dataloader, optimizer, rop,
        epochs, model_name, output_folder, visualize=0.1, device='cpu', val_freq=1
    ):
    print('\n\n------------------start training------------------------------\n\n')
    train_loss = []
    val_loss = []
    best_val_loss = 1000

    for epoch in tqdm(range(epochs)):
        model.train()
        steps = 0
        epoch_loss = 0

        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            images = batch['image'].float().to(device)
            labels = batch['label'].float().to(device)
            output = model(images)
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
                return train_loss, val_loss, bestModelPath
        #end of early stopping

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
                bestModelPath = os.path.join(output_folder,model_name+'.pt')
                torch.save(bestModel,bestModelPath)
        #log the training process
        Utils.log_to_wandb(epoch,train_loss[-1],val_loss[-1])

    print(f'finished training successfully with final validation loss of {val_loss[-1]}')
    return train_loss, val_loss, bestModelPath



def main(configFile='config.yaml'):
    monai.utils.set_determinism(seed=42)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    ### intialize from config file
    with open(configFile, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    train_label_file = cfg['training']['train_label_file']
    train_labelmap_file = cfg['model']['label_map_file']
    output_folder = cfg['training']['output_folder']
    batch_size = cfg['network']['batch_size']
    nb_epoch = cfg['network']['nb_epoch']
    cache_rate_train = cfg['training']['cache_rate_train']
    cache_rate_val = cfg['training']['cache_rate_val']
    transfer_weights = cfg['training']['transfer_weights_path']
    freezeConv = cfg['training']['freeze']
    per_slice_normalization = cfg['training']['per_slice_normalization']
    augment = cfg['training']['augment']
    randomWeightedSampling = cfg['training']['random_weighted_sampling']


    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    ##label map
    with open(train_labelmap_file) as labelmap:
        label_map = json.load(labelmap)
    N_train_classes = len(label_map.keys())

    ## setup gpu and model name
    gpu = Utils.chooseDevice(verbose=True)
    print(gpu)
    now = str(datetime.datetime.now()).replace(' ', '_')
    model_name = 'DDS_model_epochs' + str(nb_epoch) + '_time_' + now
    torch.multiprocessing.set_start_method('spawn')

    # initialize model and data
    model, optimizer, loss_function, rop = prepareModelAndCallbacks(
        N_train_classes, gpu, transfer_weights, freezeConv, per_slice_normalization
    )
    train_loader, val_loader, trainTransforms, valTransforms = prepareData(
        train_label_file=train_label_file, batch_size=batch_size, N_train_classes=N_train_classes,
        crv=cache_rate_val, crt=cache_rate_train, augment=augment, randomWeightedSampling=randomWeightedSampling,
        device=gpu
    )

    ## setup logging to wandb
    wandb.login(key=wandbkey)
    run = wandb.init(
        project='pytorch_DDS',
        name='training_'+model_name,
        config={
            'train_label_file': train_label_file,
            'label_map': label_map,
            'loss function': str(loss_function),
            'optimizer': str(optimizer),
            'train_transform': Utils.from_compose_to_list(trainTransforms),
            'val_transform': Utils.from_compose_to_list(valTransforms),
            'train_batch_size': train_loader.batch_size,
            'val_batch_size': val_loader.batch_size,
            'augmentation': augment,
            'random_weighted_sampling': randomWeightedSampling,
            'transfer_weights': transfer_weights,
            'freeze_conv': freezeConv,
            'per_slice_normalization': per_slice_normalization
        }
    )
    run_id = run.id # We remember here the run ID to be able to write the evaluation metrics

    trainloss, valloss, modelPath = train(
        model,loss_function,train_loader,val_loader,optimizer,rop,nb_epoch,
        device=gpu,val_freq=1, model_name=model_name, output_folder=output_folder
    )
    return trainloss, valloss, modelPath



if __name__=='__main__':
    #set determinism
    parser = argparse.ArgumentParser(description='This is Model training for the specified config parameters')
    parser.add_argument('-c','--configFile', action='store',metavar='c', help='pass here the config file path (from root or absolute) that should be used with your program')
    args = parser.parse_args()
    configFile = args.configFile
    trainloss, valloss, modelPath = main(configFile)
    print(f'finished model training and saved best Model to: {modelPath} \n with min training loss of {np.min(trainloss)} and min val loss of {np.min(valloss)}')



