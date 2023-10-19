import NIFTI_preparation_functions_MPR as nifp
from monai.transforms import  SaveImage
import monai
import os
import numpy as np
import multiprocessing

def preprocessImagesMonai(niftiDirec, x, y, z):
    root_dataFolder = os.path.split(niftiDirec)[0]

    NIFTI_direc = niftiDirec
    NIFIT_slices_direc = os.path.join(root_dataFolder,'NIFTI_SLICES')
    if not os.path.exists(NIFTI_direc):
        print('no nifti files found')
        return None
    if not os.path.exists(NIFIT_slices_direc):
        os.mkdir(NIFIT_slices_direc)

    print('\t\t>>>  extracting 4d info  <<<')
    images_4D_file = nifp.extract_4D_images(NIFTI_direc) #could be multiprocessed

    filepaths = []
    filenames = []
    for root, dirs, files in os.walk(NIFTI_direc):
        for i_file in files:
            if '.nii.gz' in i_file:
                filepaths.append(os.path.join(root,i_file))
                filenames.append(i_file.split('.nii.gz')[0])
    print('\t\t>>>  create data set  <<<')
    print(f'{len(filepaths)} nifti images found in the dataset')
    dataset = [{'image':filepath,'label':filename} for filepath, filename in zip(filepaths,filenames)]
    

    class SaveIndividualSlices(monai.transforms.MapTransform):
        def __init__(self, keys, targetDir, split):
            self.keys = keys
            self.targetDir = targetDir
            if split:
                self.outdir = os.path.join(self.targetDir,split)
            else:
                self.outdir = self.targetDir

        def __call__(self, data):
            os.makedirs(self.targetDir,exist_ok=True)
            depth = data[self.keys[0]].shape[-1]
            for i in range(depth):
                saver = SaveImage(
                    output_dir=self.outdir,
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
                slice = slice.reshape((1,x,y,1))
                metadata = dict(data[f'{self.keys[0]}_meta_dict'])
                try:
                    saver(img=slice,meta_data=metadata)
                except RuntimeError:
                    print(f'rutime error when trying to save image {data[self.keys[1]]} to slice{i}')
                    break
            return data

    dataTransform = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=['image'], image_only=False, ensure_channel_first=False),
            monai.transforms.EnsureChannelFirstd(keys=['image']),
            monai.transforms.Orientationd(keys=['image'],axcodes='LAS'),
            monai.transforms.Resized(keys=['image'], spatial_size=[x, y, z]),
            monai.transforms.ScaleIntensityd(keys=['image']),
            SaveIndividualSlices(keys=['image', 'label'], targetDir=NIFIT_slices_direc,split=None)
        ]
    )

    ds = monai.data.Dataset(dataset, dataTransform)
    dl = monai.data.DataLoader(ds, batch_size=8, num_workers=multiprocessing.cpu_count()-2)
    print('\t\t>>>  applying transforms and saving slices <<<')
    counter = -1
    # for i in tqdm(dl):
    #     counter+=1
    #     continue
    iterloader = iter(dl)
    for i in range(len(iterloader)):
        try:
            _ = next(iterloader)
        except:
            print(f'could not convert image {i}')
    print('\t\t>>>  create labels  <<<')
    nifp.create_label_file(NIFIT_slices_direc,root_dataFolder, images_4D_file,'Labels.txt')
    return NIFIT_slices_direc
