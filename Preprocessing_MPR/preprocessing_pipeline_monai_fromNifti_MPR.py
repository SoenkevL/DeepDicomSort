import os
import yaml
import DICOM_preparation_functions_MPR as DPF
import NIFTI_preparation_functions_MPR as NPF
import preprocessingFunctionMonai_MPR as PFM
import time
import argparse

parser = argparse.ArgumentParser(description='This is the preprocessing pipeline for a nifti folder depening on what is specified in the config.yaml file.')
parser.add_argument('-c','--configFile', action='store',metavar='c', help='pass here the config file path (from root or absolute) that should be used with your program')
parser.add_argument('-n','--nifti_folder',action='store',required=True, help='pass here the filepath to the nifti folder')
args = parser.parse_args()
start_time = time.time()
with open(args.configFile, 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

x_image_size = cfg['data_preparation']['image_size_x']
y_image_size = cfg['data_preparation']['image_size_y']
z_image_size = cfg['data_preparation']['image_size_z']
train_test_split = cfg['data_preparation']['train_test_split']


print('applying monai transforms and splitting images')
nifti_slices_folder = PFM.preprocessImagesMonai(args.nifti_folder,x_image_size,y_image_size,z_image_size,train_test_split=train_test_split)

elapsed_time = time.time() - start_time

print(elapsed_time)