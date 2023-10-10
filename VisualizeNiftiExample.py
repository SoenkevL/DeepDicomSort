import os
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt


certaintyThreshhold = 1
dataframe_path = '/trinity/home/r098375/DDS/DeepDicomSort/.testing/combination1_a2_a3_o3/Predictions_DDS_model_epochs100_time_2023-10-09_15:13_ensamblePredictions.csv'
imagePath = os.path.join(os.path.dirname(dataframe_path),'visualizations')
if not os.exists(imagePath):
    os.mkdir(imagePath)
FullResultFrame = pd.read_csv(dataframe_path)
#only visualize examples where vote and prediction dont match
FullResultFrame = FullResultFrame[FullResultFrame['certainty']>=certaintyThreshhold]
FullResultFrame = FullResultFrame[FullResultFrame['vote']-FullResultFrame['groundTruth']!=0]
for idx in FullResultFrame.index:
    img_path = FullResultFrame['imageID'][idx]
    nimg = nib.load(img_path)
    data = nimg.get_fdata()
    affine = nimg.affine
    header = nimg.header
    plt.figure()
    plt.imshow(data[:,:],cmap='gray')
    plt.title(f"{FullResultFrame['NIFTI_name'][idx]}\n was voted {FullResultFrame['vote'][idx]} and is actually {FullResultFrame['groundTruth'][idx]}")
    plt.savefig(os.path.join(imagePath,f'p{idx}.png'))
    plt.close('all')
#%%
