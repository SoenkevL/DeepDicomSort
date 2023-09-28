import os
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt


dataframe_path = '/trinity/home/r098375/DDS/XNAT/Predictions/Predictions_DDS_model_epochs100_time_2023-09-21_16:10_ensamblePredictions.csv'
imagePath = os.path.join(os.path.dirname(dataframe_path),'visualizations')
dataframe = pd.read_csv(dataframe_path).head(10)
for idx in range(len(dataframe)):
    img_path = dataframe['imageID'][idx]
    nimg = nib.load(img_path)
    data = nimg.get_fdata()
    affine = nimg.affine
    header = nimg.header
    plt.figure()
    plt.imshow(data[:,:],cmap='gray')
    plt.title(f"{dataframe['imageID'][idx]} : {dataframe['prediction'][idx]}")
    plt.savefig(os.path.join(imagePath,f'p{idx}.png'))
#%%
