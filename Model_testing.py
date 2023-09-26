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


def load_data(test_label_file):
    test_image_IDs, test_image_labels, _, extra_inputs = Utils.load_labels(test_label_file)

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

def testing(model, test_loader, device, ResultsFrame, output_folder, model_name):
    for datapoint in tqdm(test_loader):
        image = datapoint['image'].to(device=device)
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
    while os.path.exists(out_file):
        out_file = os.path.join(output_folder, f'Predictions_copy{fileCounter}_' + model_name + '.csv')
        fileCounter+=1
        if fileCounter==5:
            print('too many copies copy 5 will be overriden')
            break
    ResultsFrame.to_csv(out_file,index=False)
    return ResultsFrame



def main(configFile='config.yaml'):
    with open(configFile, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    test_label_file = cfg['testing']['test_label_file']
    model_file = cfg['model']['model_file']
    output_folder = cfg['testing']['output_folder']

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    gpu = Utils.chooseDevice(verbose=True)
    model = torch.load(f=model_file,map_location=gpu)
    model = model.to(device=gpu)
    model.eval()

    model_name = os.path.basename(os.path.normpath(model_file)).split('.pt')[0]

    ResultsFrame_empty, test_loader = load_data(test_label_file)

    ResultsFrame = testing(model,test_loader,gpu,ResultsFrame_empty,output_folder,model_name)

    

    
if __name__=='__main__':
    main() #can specify file to different config than standard 'config.yaml' here as input argument
