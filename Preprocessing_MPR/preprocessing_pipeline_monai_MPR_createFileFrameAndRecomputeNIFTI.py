import os
import yaml
import DICOM_preparation_functions_MPR_noMovingOrCopying as DPF
import NIFTI_preparation_functions_MPR as NPF
import preprocessingFunctionMonai_MPR as PFM
import time
import argparse
import shutil

def protectConfig(configFile):
    filename = os.path.basename(configFile)
    filepath = os.path.dirname(configFile)
    filenameCopy = filename.split('.yaml')[0]+'_copy.yaml'
    configFile_copy = os.path.join(filepath, filenameCopy)
    shutil.copy(configFile, configFile_copy)
    return configFile_copy

def createDicomHeaderInfoCsv(DatasetPath):
    dicomPath = DatasetPath+'/DICOM_STRUCTURED'
    outPath = DatasetPath+'/DicomHeaderFrame.csv'
    dicts = []
    for root, folders, files in os.walk(dicomPath):
        if files:
            dicomAttributeDict = {
                'structuredPath':None,
                'InstanceCreationDate':None,
                'Manufacturer':None,
                'Modality':None,
                'SeriesDescription':None,
                'PatientID':None,
                'ProtocolName':None,
                'StudyInstanceUID':None,
                'SeriesInstanceUID':None,
                'RepetitionTime':None,
                'EchoTime':None,
                'InversionTime':None
            }
            data = pdicom.read_file(os.path.join(root, files[0]), stop_before_pixels=True)
            dicomAttributeDict['structuredPath'] = root
            try:
                dicomAttributeDict['InstanceCreationDate'] = data.InstanceCreationDate
            except:
                pass
            try:
                dicomAttributeDict['Manufacturer'] = data.Manufacturer
            except:
                pass
            try:
                dicomAttributeDict['Modality'] = data.Modality
            except:
                pass
            try:
                dicomAttributeDict['SeriesDescription'] = data.SeriesDescription
            except:
                pass
            try:
                dicomAttributeDict['PatientID'] = data.PatientID
            except:
                pass
            try:
                dicomAttributeDict['ProtocolName'] = data.ProtocolName
            except:
                pass
            try:
                dicomAttributeDict['StudyInstanceUID'] = data.StudyInstanceUID
            except:
                pass
            try:
                dicomAttributeDict['SeriesInstanceUID'] = data.SeriesInstanceUID
            except:
                pass
            try:
                dicomAttributeDict['RepetitionTime'] = data.RepetitionTime
            except:
                pass
            try:
                dicomAttributeDict['EchoTime'] = data.EchoTime
            except:
                pass
            try:
                dicomAttributeDict['InversionTime'] = data.InversionTime
            except:
                pass
            dicts.append(dicomAttributeDict)
    df_dicom = pd.DataFrame(dicts)
    df_dicom.to_csv(outPath, index=False)
    return outPath

parser = argparse.ArgumentParser(description='This is the preprocessing pipeline for a data or nifti folder depening on what is specified in the config.yaml file.')
parser.add_argument('-c','--configFile', action='store',metavar='c', help='pass here the config file path (from root or absolute) that should be used with your program')
parser.add_argument('-s', '--skip_sorting', action='store_true', help='set this flag to skip the sorting and splitting in series step of the preprocessing pipeline')
args = parser.parse_args()
start_time = time.time()
config = protectConfig(args.configFile)
with open(config, 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

x_image_size = cfg['data_preparation']['image_size_x']
y_image_size = cfg['data_preparation']['image_size_y']
z_image_size = cfg['data_preparation']['image_size_z']
DICOM_FOLDER = cfg['preprocessing']['root_dicom_folder']
df_path = cfg['preprocessing']['df_path']
print(f'preprocessing {DICOM_FOLDER} without moving or copying dicom files to create the linking dataframes')

DEFAULT_SIZE = [x_image_size, y_image_size, z_image_size]


def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def is_odd(number):
    return number % 2 != 0


print(f'number of elements in dicom folder:{len(os.listdir(DICOM_FOLDER))}')
print('Sorting DICOM to structured folders....')
if not args.skip_sorting:
    structured_dicom_folder = DPF.sort_DICOM_to_structured_folders(DICOM_FOLDER, df_path)
else:
    structured_dicom_folder = os.path.dirname(DICOM_FOLDER)+'/DICOM_STRUCTURED'

# Turn the following step on if you have problems running the pipeline
# It will replaces spaces in the path names, which can sometimes
# Cause errors with some tools
# print('Removing spaces from filepaths....')
# DPF.make_filepaths_safe_for_linux(structured_dicom_folder)
#
print('Checking and splitting for double scans in folders....')
DPF.split_in_series(structured_dicom_folder, df_path)

print('Converting DICOMs to NIFTI....')
nifti_folder = NPF.convert_DICOM_to_NIFTI_monai(structured_dicom_folder, df_path)

print('applying monai transforms and splitting images')
nifti_slices_folder = PFM.preprocessImagesMonai(nifti_folder,x_image_size,y_image_size,z_image_size)

print('creating DICOM frame')
datapath = os.path.dirname(DICOM_FOLDER)
createDicomHeaderInfoCsv(datapath)

elapsed_time = time.time() - start_time

print(elapsed_time)
