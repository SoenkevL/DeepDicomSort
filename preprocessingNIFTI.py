from Preprocessing import NIFTI_preparation_functions as nifp
import yaml
from monai.transforms import  SaveImage
from tqdm import tqdm
import monai
import os
import numpy as np
from sklearn.model_selection import train_test_split

with open('config_BRATS.yaml', 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

x_image_size = cfg['data_preparation']['image_size_x']
y_image_size = cfg['data_preparation']['image_size_y']
z_image_size = cfg['data_preparation']['image_size_z']
DICOM_FOLDER = cfg['preprocessing']['root_dicom_folder']

root_dataFolder = os.path.split(DICOM_FOLDER)[0]

NIFTI_direc = os.path.join(root_dataFolder,'NIFTI')
NIFIT_slices_direc = os.path.join(root_dataFolder,'NIFTI_SCLICES')
if not os.path.exists(NIFTI_direc):
    os.mkdir(NIFTI_direc)
if not os.path.exists(NIFIT_slices_direc):
    os.mkdir(NIFIT_slices_direc)

print('>>> extracting 4d info')
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
train_dataset, test_dataset = train_test_split(dataset,random_state=42,shuffle=True,test_size=0.1)

class SaveIndividualSlices(monai.transforms.MapTransform):
    def __init__(self, keys, targetDir, split):
        self.keys = keys
        self.targetDir = targetDir
        self.split = split

    def __call__(self, data):
        os.makedirs(self.targetDir,exist_ok=True)
        depth = data[self.keys[0]].shape[-1]
        for i in range(depth):
            saver = SaveImage(
                output_dir=os.path.join(self.targetDir,self.split),
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
            slice = slice.reshape((1,x_image_size,y_image_size,1))
            metadata = dict(data[f'{self.keys[0]}_meta_dict'])
            saver(img=slice,meta_data=metadata)
            # filename = str(data[self.keys[1]])
            # savepath = os.path.join(self.targetDir,f'{filename}__s{i}.nii.gz')
            # affine = metadata['affine']
            # nib.save(img=nib.Nifti1Pair(slice,affine=affine), filename=savepath)
        return data



trainTransform = monai.transforms.Compose(
    [
        monai.transforms.LoadImaged(keys=['image'], image_only=False, ensure_channel_first=False),
        monai.transforms.EnsureChannelFirstd(keys=['image']),
        monai.transforms.Orientationd(keys=['image'],axcodes='LAS'),
        monai.transforms.Resized(keys=['image'], spatial_size=[x_image_size, y_image_size, z_image_size]),
        monai.transforms.ScaleIntensityd(keys=['image']),
        SaveIndividualSlices(keys=['image', 'label'], targetDir=NIFIT_slices_direc,split='train')
    ]
)
testTransform = monai.transforms.Compose(
    [
        monai.transforms.LoadImaged(keys=['image'], image_only=False, ensure_channel_first=False),
        monai.transforms.EnsureChannelFirstd(keys=['image']),
        monai.transforms.Orientationd(keys=['image'],axcodes='LAS'),
        monai.transforms.Resized(keys=['image'], spatial_size=[x_image_size, y_image_size, z_image_size]),
        monai.transforms.ScaleIntensityd(keys=['image']),
        SaveIndividualSlices(keys=['image', 'label'], targetDir=NIFIT_slices_direc,split='test')
    ]
)

ds_train = monai.data.Dataset(train_dataset, trainTransform)
dl_train = monai.data.DataLoader(ds_train, batch_size=1, num_workers=1)
ds_test = monai.data.Dataset(test_dataset, testTransform)
dl_test = monai.data.DataLoader(ds_test, batch_size=1, num_workers=1)

print('>>> create train set')
counter = -1
try:
    for i in tqdm(dl_train):
        counter+=1
        continue
except:
    print(i['label'])
print('>>> create test set')
counter = -1
try:
    for i in tqdm(dl_test):
        counter+=1
        continue
except:
    print(i['label'])

nifp.create_label_file(os.path.join(NIFIT_slices_direc,'train'), images_4D_file,'trainLabelsBRATS.txt')
nifp.create_label_file(os.path.join(NIFIT_slices_direc,'test'), images_4D_file,'testLabelsBRATS.txt')