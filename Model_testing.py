import pandas as pd
import monai
import os
import torch
import numpy as np
import yaml
import datetime
from tqdm import tqdm
import Pytorch_monai.Model_and_transforms as MF
import Pytorch_monai.Utils as Utils
import json
import argparse
import Model_analysis as MA


def createMetaInfoDict(modelFile, test_label_file, train_label_file, labelmap):
    return {
        'model': os.path.basename(modelFile),
        'train_file': train_label_file,
        'test_file': test_label_file,
        'labelmap': labelmap
    }

def load_data(test_label_file, labelmap):
    test_image_IDs, test_image_labels, _, extra_inputs = Utils.load_labels(test_label_file,nb_classes=len(labelmap) )

    testTransforms = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=['image'],image_only=True,reader='NibabelReader'),
            monai.transforms.EnsureTyped(keys=['image']),
            monai.transforms.EnsureChannelFirstd(keys=['image'])
        ]
    )

    test_data_dict = [{"image":image_name,"label":label,"ID":image_name} for image_name, label in zip(test_image_IDs,test_image_labels)]
    test_ds = monai.data.CacheDataset(data=test_data_dict,transform=testTransforms,cache_rate=0.5,num_workers=4)
    test_loader = monai.data.DataLoader(test_ds,shuffle=False,num_workers=0,batch_size=1)
    # do some checks on the labels
    testLabelList = [np.argmax(dictItem['label']) for dictItem in test_data_dict]

    rawCols = [f'raw_{i}' for i in range(len(np.unique(test_image_labels)))]
    predictCols = ['imageID','groundTruth','prediction']
    cols = predictCols+rawCols
    ResultsFrame = pd.DataFrame(columns=cols)
    return ResultsFrame, test_loader

def testing(model, test_loader, device, ResultsFrame, output_folder, model_name, meta_dict):
    '''
    creates a dataframe which stores the results of the model
    '''
    for datapoint in tqdm(test_loader):
        image = datapoint['image'].float().to(device=device)
        groundTruth = np.argmax(datapoint['label']).to('cpu').detach().numpy()
        ID = datapoint['ID']
        prediction_raw = model(image)
        prediction = np.argmax(prediction_raw)
        predDict={'imageID':ID,'groundTruth':groundTruth,'prediction':prediction}
        predictionRawNp = prediction_raw.to('cpu').detach().numpy().squeeze(0)
        rawDict = {f'raw_{i}':prob for i,prob in enumerate(predictionRawNp)}
        Dict = predDict
        Dict.update(rawDict)
        result = pd.DataFrame(Dict)
        if ResultsFrame.empty:
            ResultsFrame = result
        else:
            ResultsFrame = pd.concat([ResultsFrame,result],ignore_index=True)



    fileCounter=0
    out_file = os.path.join(output_folder, 'Predictions_' + model_name + '.csv')
    out_meta_file = os.path.join(output_folder, 'Predictions_' + model_name + '_metaDict.json')
    while os.path.exists(out_file):
        out_file = os.path.join(output_folder, f'Predictions_copy{fileCounter}_' + model_name + '.csv')
        out_meta_file = os.path.join(output_folder, f'Predictions_copy{fileCounter}_' + model_name + '_metaDict.json')
        fileCounter+=1
        if fileCounter==5:
            print('too many copies copy 5 will be overriden')
            break
    ResultsFrame.to_csv(out_file,index=False)
    with open(out_meta_file,'w') as f:
        json.dump(meta_dict,f)
    return out_file



def main(configFile):
    with open(configFile, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    train_label_file = cfg['training']['train_label_file']
    test_label_file = cfg['testing']['test_label_file']
    output_folder = cfg['testing']['output_folder']
    model_file = cfg['model']['model_file']
    label_map_file = cfg['model']['label_map_file']

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        
    with open(label_map_file) as Lmap:
        label_map = json.load(Lmap)

    label_map = dict(label_map)
    print(len(label_map))
    metaDict = createMetaInfoDict(modelFile=model_file, test_label_file=test_label_file, train_label_file=train_label_file, labelmap=label_map)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    gpu = Utils.chooseDevice(verbose=True)
    model = torch.load(f=model_file,map_location=gpu)
    model = model.to(device=gpu)
    model.eval()

    model_name = os.path.basename(os.path.normpath(model_file)).split('.pt')[0]

    ResultsFrame_empty, test_loader = load_data(test_label_file, label_map)

    out_file = testing(model,test_loader,gpu,ResultsFrame_empty,output_folder,model_name,meta_dict=metaDict)
    return out_file
    

    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='This is Model test for the specified config parameters')
    parser.add_argument('-c','--configFile', action='store',metavar='c', help='pass here the config file path (from root or absolute) that should be used with your program')
    args = parser.parse_args()
    configFile = args.configFile
    outfile = main(configFile)
    print('finished testing')
    MA.main(outfile, testing=True, certainties='[1, 0.8, 0.6]')
