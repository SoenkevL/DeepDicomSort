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

model_file = '.training/out/DDS_model_epochs100_time_2023-09-21_16:16:19.587076.pt'
batch_size = 1

with open('config_BRATS.yaml', 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

gpu = Utils.chooseDevice(verbose=True)

test_label_file = cfg['testing']['test_label_file_pruned']
output_folder = cfg['testing']['output_folder']
x_image_size = cfg['data_preparation']['image_size_x']
y_image_size = cfg['data_preparation']['image_size_y']
model = torch.load(f=model_file,map_location=gpu)

model_name = os.path.basename(os.path.normpath(model_file)).split('.pt')[0]
out_file = os.path.join(output_folder, 'Predictions_' + model_name + '.csv')

model = model.to(device=gpu)
model.eval()

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
for datapoint in tqdm(test_loader):
    image = datapoint['image'].to(device=gpu)
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
while os.path.exists(out_file):
    out_file = os.path.join(output_folder, f'Predictions_copy{fileCounter}_' + model_name + '.csv')
    fileCounter+=1
    if fileCounter==5:
        print('too many copies')
        break
ResultsFrame.to_csv(out_file,index=False)