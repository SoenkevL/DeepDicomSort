from Preprocessing import NIFTI_preparation_functions as nifp
import yaml
from monai.transforms import  SaveImage
from tqdm import tqdm
import shutil
import monai
import os
import numpy as np

with open('./config.yaml', 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

x_image_size = cfg['data_preparation']['image_size_x']
y_image_size = cfg['data_preparation']['image_size_y']
z_image_size = cfg['data_preparation']['image_size_z']
DICOM_FOLDER = cfg['preprocessing']['root_dicom_folder']
DCM2NIIX_BIN = cfg['preprocessing']['dcm2niix_bin']
FSLREORIENT_BIN = cfg['preprocessing']['fslreorient2std_bin']
FSLVAL_BIN = cfg['preprocessing']['fslval_bin']

root_dataFolder = os.path.split(DICOM_FOLDER)[0]

NIFTI_direc = os.path.join(root_dataFolder,'NIFTI')
NIFIT_slices_direc = os.path.join(root_dataFolder,'NIFTI_SCLICES')
if not os.path.exists(NIFTI_direc):
    os.mkdir(NIFTI_direc)
if not os.path.exists(NIFIT_slices_direc):
    os.mkdir(NIFIT_slices_direc)

images_4D_file = nifp.extract_4D_images(NIFTI_direc)

filepaths = []
filenames = []
for root, dirs, files in os.walk(NIFTI_direc):
    for i_file in files:
        if '.nii.gz' in i_file:
            filepaths.append(os.path.join(root,i_file))
            filenames.append(i_file.split('.nii.gz')[0])
print(len(filepaths))
dataset = [{'image':filepath,'label':filename} for filepath, filename in zip(filepaths,filenames)]

class SaveIndividualSlices(monai.transforms.MapTransform):
    def __init__(self, keys, targetDir):
        self.keys = keys
        self.targetDir = targetDir

    def __call__(self, data):
        os.makedirs(self.targetDir,exist_ok=True)
        depth = data[self.keys[0]].shape[-1]
        for i in range(depth):
            saver = SaveImage(
                output_dir=self.targetDir,
                output_postfix=f'_s{i}',
                output_ext='.nii.gz',
                resample=False,
                squeeze_end_dims=True,
                data_root_dir=NIFTI_direc,
                separate_folder=False,
                print_log=False,
                savepath_in_metadict=True,
            )
            slice = data[self.keys[0]][:,:,:,i]
            slice = np.array(slice)
            slice = slice.reshape((1,256,256,1))
            metadata = dict(data[f'{self.keys[0]}_meta_dict'])
            saver(img=slice,meta_data=metadata)
        return data



baseTransform = monai.transforms.Compose(
    [
        monai.transforms.LoadImaged(keys=['image'], image_only=False, ensure_channel_first=False),
        monai.transforms.EnsureChannelFirstd(keys=['image']),
        monai.transforms.Orientationd(keys=['image'],axcodes='LAS'),
        monai.transforms.Resized(keys=['image'], spatial_size=[256, 256, 25]),
        monai.transforms.ScaleIntensityd(keys=['image']),
        SaveIndividualSlices(keys=['image', 'label'], targetDir=NIFIT_slices_direc)
    ]
)

ds_2d = monai.data.Dataset(dataset, baseTransform)
dl_2d = monai.data.DataLoader(ds_2d, batch_size=1, num_workers=1)

counter = -1
try:
    for i in tqdm(dl_2d):
        counter+=1
        continue
except:
    print(i['label'])

nifp.create_label_file(NIFIT_slices_direc,images_4D_file)