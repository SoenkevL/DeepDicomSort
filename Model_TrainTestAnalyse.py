import Model_training_new as MTR
import Model_testing as MTE
import Model_analysis as MA
from Pytorch_monai.Utils import protectConfig
import yaml
import argparse

# commandline arguments
parser = argparse.ArgumentParser(description='This is a full model testing pipeine. In the config file atleast the'
                                             ' following things need to be specified: \n'
                                             'all training and testing parameters aswell as the labelmap.'
                                             ' The model will be added automatically during the script')
parser.add_argument('-c', '--configFile', action='store', required=True, help='pass here the config file path'
                                                                            ' (from root or absolute) that should'
                                                                            ' be used with your program')
args = parser.parse_args()
configFile = protectConfig(args.configFile)
print('start training')
_, _, modelPath = MTR.main(configFile)
with open(configFile, 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)
cfg['model']['model_file'] = str(modelPath)
with open(configFile, 'w') as ymlfile:
    yaml.safe_dump(cfg, ymlfile)
print('finished training, running test file now')
predictions = MTE.main(configFile)
print('finished predicting, analysing the results now')
MA.main(predictions, testing=True, certainties='[1, 0.8, 0.6]', vis=True)
print('analysed results and created relevant outputs, pipeline finished')
