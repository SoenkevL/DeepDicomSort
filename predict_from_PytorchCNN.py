import numpy as np
import Tools.data_IO as data_IO
import torch
import monai
import yaml
import os
import pandas as pd
import Pytorch_monai.Utils as Utils

model_file = '.training/out/DDS_model_epochs100_time_2023-09-19_13:13:03.305171.pt'
batch_size = 1

with open('./config.yaml', 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

test_label_file = cfg['testing']['test_label_file']
output_folder = cfg['testing']['output_folder']
x_image_size = cfg['data_preparation']['image_size_x']
y_image_size = cfg['data_preparation']['image_size_y']

gpu = Utils.chooseDevice()

model_name = os.path.basename(os.path.normpath(model_file)).split('.pt')[0]
out_file = os.path.join(output_folder, 'Predictions_' + model_name + '.csv')


test_image_IDs, test_image_labels, _, extra_inputs = Utils.load_labels(test_label_file)
test_image_IDs, test_image_labels = test_image_IDs[110000:111000], test_image_labels[110000:111000] #for the small models this is not inside the training data

testTransforms = monai.transforms.Compose(
    [
        monai.transforms.LoadImaged(keys=['image'],image_only=True,reader='NibabelReader'),
        monai.transforms.EnsureTyped(keys=['image']),
        monai.transforms.EnsureChannelFirstd(keys=['image'],channel_dim=0)
    ]
)

#create data dicitionaries and data loader to which monai transforms can be applied (transforms stored in Model_and_transforms.py)
test_data_dict = [{"image":image_name,"label":label,"ID":image_name} for image_name, label in zip(test_image_IDs,test_image_labels)]
test_ds = monai.data.CacheDataset(data=test_data_dict,transform=testTransforms,cache_rate=1,num_workers=4,progress=True)
test_loader = monai.data.DataLoader(test_ds,batch_size=batch_size,shuffle=False,num_workers=0)
#intialize model, optimizer, loss
model = torch.load(f=model_file).to(device=gpu)
model.eval()

rawCols = [f'raw_{i}' for i in range(len(np.unique(test_image_labels)))]
predictCols = ['imageID','groundTruth','prediction']
cols = predictCols+rawCols
ResultsFrame = pd.DataFrame(columns=cols)
for datapoint in test_loader:
    image = datapoint['image'].to(device=gpu)
    groundTruth = np.argmax(datapoint['label'],axis=1)
    ID = datapoint['ID']
    prediction_raw = model(image)
    print(prediction_raw)
    prediction = np.argmax(prediction_raw,axis=1)
    predDict={'imageID':ID,'groundTruth':groundTruth,'prediction':prediction}
    predictionRawNp = prediction_raw.to('cpu').detach().numpy()
    print([i for i in predictionRawNp])
    rawDict = {f'raw_{i}':prob for i,prob in enumerate(predictionRawNp)}
    Dict = predDict
    Dict.update(rawDict)
    print(Dict)
    result = pd.DataFrame(Dict)
    ResultsFrame = pd.concat([ResultsFrame,result],ignore_index=True)

ResultsFrame.to_csv(out_file)
