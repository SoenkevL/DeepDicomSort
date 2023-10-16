import os
import Model_predicting as Mp
import Model_SortToBids as Msb
import os
import yaml
import Preprocessing.DICOM_preparation_functions as DPF
import Preprocessing.NIFTI_preparation_functions as NPF
import Preprocessing.preprocessingFunctionMonai as PFM
import time
import argparse

parser = argparse.ArgumentParser(description='This is the preprocessing pipeline for a data or nifti folder depening on what is specified in the config.yaml file.')
parser.add_argument('-c','--configFile', action='store',metavar='c', help='pass here the config file path (from root or absolute) that should be used with your program')
args = parser.parse_args()
start_time = time.time()
with open(args.configFile, 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

x_image_size = cfg['data_preparation']['image_size_x']
y_image_size = cfg['data_preparation']['image_size_y']
z_image_size = cfg['data_preparation']['image_size_z']
DICOM_FOLDER = cfg['preprocessing']['root_dicom_folder']
train_test_split = cfg['data_preparation']['train_test_split']
print(f'preprocessing {DICOM_FOLDER}')

DEFAULT_SIZE = [x_image_size, y_image_size, z_image_size]


def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def is_odd(number):
    return number % 2 != 0


print(f'number of elements in dicom folder:{len(os.listdir(DICOM_FOLDER))}')
print('Sorting DICOM to structured folders....')
structured_dicom_folder = DPF.sort_DICOM_to_structured_folders(DICOM_FOLDER)

# Turn the following step on if you have problems running the pipeline
# It will replaces spaces in the path names, which can sometimes
# Cause errors with some tools
# print('Removing spaces from filepaths....')
# DPF.make_filepaths_safe_for_linux(structured_dicom_folder)
#
print('Checking and splitting for double scans in folders....')
DPF.split_in_series(structured_dicom_folder)

print('Converting DICOMs to NIFTI....')
nifti_folder = NPF.convert_DICOM_to_NIFTI(structured_dicom_folder)

print('applying monai transforms and splitting images')
nifti_slices_folder = PFM.preprocessImagesMonai(nifti_folder,x_image_size,y_image_size,z_image_size,train_test_split=train_test_split)
cfg['post_processing']['prediction_folder'] = nifti_slices_folder

with open(args.configFile,'w') as ymlfile:
    yaml.safe_dump(cfg, ymlfile)
elapsed_time = time.time() - start_time
print(elapsed_time)
print('finished preprocessing, start predicting')


start_time = time.time()
predictionFile = Mp.main(args.configFile) #I could use here model testing with testing=false if I want to go from the empty label file instead of the folder
cfg['post_processing']['prediction_file'] = predictionFile
with open(args.configFile,'w') as ymlfile:
    yaml.safe_dump(cfg, ymlfile)
elapsed_time = time.time() - start_time
print(elapsed_time)
print('finished predicting, moving files')
Msb.main(args.configFile) #this is still pretty much from the original so it needs some tweeking to be inline with the rest
print('all files moved, pipeline finished successfully')




