import Model_predicting as Mp
import yaml
import Preprocessing_cleaned.preprocessing_pipeline_monai as prep_pipe
import time
import argparse
from Pytorch_monai.Utils import  protectConfig
'''
This file is used as a script to run
-preprocessing
-predicting
-analysing
'''
parser = argparse.ArgumentParser(description='This is the preprocessing pipeline for a data or nifti folder depening on what is specified in the config.yaml file.')
parser.add_argument('-c','--configFile', action='store', required=True, help='pass here the config file path (from root or absolute) that should be used with your program')
args = parser.parse_args()
start_time = time.time()
config = protectConfig(args.configFile)
with open(config, 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)
print('start preprocessing pipeline')
nsf = prep_pipe.full_preprocessing(config)
cfg['post_processing']['prediction_folder'] = nsf
with open(config,'w') as ymlfile:
    yaml.safe_dump(cfg, ymlfile)
print('finished preprocessing pipeline, start predicting')


predictionFile = Mp.main(config) #I could use here model testing with testing=false if I want to go from the empty label file instead of the folder
cfg['post_processing']['prediction_file'] = predictionFile
with open(config,'w') as ymlfile:
    yaml.safe_dump(cfg, ymlfile)
print('finished predicting')
elapsed_time = time.time() - start_time
print(elapsed_time)
# Msb.main(args.configFile) #this is still pretty much from the original so it needs some tweeking to be inline with the rest
# print('all files moved, pipeline finished successfully')




