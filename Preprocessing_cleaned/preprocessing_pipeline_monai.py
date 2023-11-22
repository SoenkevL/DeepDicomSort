'''
a script that takes care of the preprocessing of raw dicom folder to structure them, convert them into nifit files
extract 4d information, resample them to the specified sizes in the config respectively, scale their volume intensities
and finally split each nifti volume into 25 2d slices
'''
import os
import yaml
import Preprocessing_cleaned.prep_functions as prep
import time
import argparse


def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def is_odd(number):
    return number % 2 != 0


def full_preprocessing(config):
    '''
    runs the full preprocessing pipeline using the provided config
    dicom sorting - dicom to nifti conversion - nifti volume to slices
    Logs the whole process in the csv provided in the config
    requires all preprocessing fields to be set inside the config
    '''
    start_time = time.time()
    with open(config, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    x_image_size = cfg['data_preparation']['image_size_x']
    y_image_size = cfg['data_preparation']['image_size_y']
    z_image_size = cfg['data_preparation']['image_size_z']
    DICOM_FOLDER = cfg['preprocessing']['root_dicom_folder']
    df_path = cfg['preprocessing']['df_path']
    print(f'preprocessing {DICOM_FOLDER}')

    if not os.path.exists(df_path):
        os.makedirs(df_path, exist_ok=True)
    if os.path.splitext(df_path)[1]:
        pass
    else:
        df_path = os.path.join(df_path,'FileFrame.csv')

    print(f'number of elements in dicom folder:{len(os.listdir(DICOM_FOLDER))}')
    print('Sorting DICOM to structured folders....')
    structured_dicom_folder = prep.sort_DICOM_to_structured_folders(DICOM_FOLDER, df_path)

    print('Checking and splitting for double scans in folders....')
    prep.split_in_series(structured_dicom_folder,df_path)

    print('Converting DICOMs to NIFTI....')
    nifti_folder = prep.convert_DICOM_to_NIFTI_monai(structured_dicom_folder, df_path)

    print('applying monai transforms and splitting images')
    nifti_slices_folder = prep.preprocessImagesMonai(nifti_folder,x_image_size,y_image_size,z_image_size, df_path)

    print('creating dicom header info dataframe')
    datapath = os.path.dirname(df_path)
    prep.createDicomHeaderInfoCsv(datapath)

    elapsed_time = time.time() - start_time

    print(f'running the full pipeline took {elapsed_time}s')
    return nifti_slices_folder

def dicom_sorting(config):
    '''
    does only the dicom sorting without any nifti processing steps
    '''
    start_time = time.time()
    with open(config, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    DICOM_FOLDER = cfg['preprocessing']['root_dicom_folder']
    df_path = cfg['preprocessing']['df_path']
    print(f'preprocessing {DICOM_FOLDER}')

    if os.path.splitext(df_path)[1]:
        pass
    else:
        df_path = os.path.join(df_path,'FileFrame.csv')

    print(f'number of elements in dicom folder:{len(os.listdir(DICOM_FOLDER))}')
    print('Sorting DICOM to structured folders....')
    structured_dicom_folder = prep.sort_DICOM_to_structured_folders(DICOM_FOLDER, df_path)

    print('Checking and splitting for double scans in folders....')
    prep.split_in_series(structured_dicom_folder,df_path)

    elapsed_time = time.time() - start_time

    print(f'dicom sorting took {elapsed_time}s')

def dicom_to_nifti_from_structured(config):
    '''
    does only the nifti conversion from "STRUCTURED_DICOM" folder
    '''
    print('Converting DICOMs to NIFTI....')
    start_time = time.time()
    with open(config, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    DICOM_FOLDER = cfg['preprocessing']['root_dicom_folder']
    df_path = cfg['preprocessing']['df_path']

    if os.path.splitext(df_path)[1]:
        pass
    else:
        df_path = os.path.join(df_path,'FileFrame.csv')

    structured_dicom_folder = os.path.join(os.path.dirname(DICOM_FOLDER),'DICOM_STRUCTURED')
    if not os.path.exists(structured_dicom_folder):
        print('no structured dicom folder exists, either run the whole preprocessing or move your structured dicoms \
              into a folder called "STRUCTURED_DICOM"')
        return None
    else:
        nifti_folder = prep.convert_DICOM_to_NIFTI_monai(structured_dicom_folder, df_path)
        print('finished conversion')
        elapsed_time = time.time() - start_time
        print(f'convertig dicom to nifti took {elapsed_time}s')
        return nifti_folder

def nifti_processing(config):
    '''
    does only the nifti processing from the "NIFTI" folder
    '''
    with open(config, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)


    x_image_size = cfg['data_preparation']['image_size_x']
    y_image_size = cfg['data_preparation']['image_size_y']
    z_image_size = cfg['data_preparation']['image_size_z']
    DICOM_FOLDER = cfg['preprocessing']['root_dicom_folder']
    df_path = cfg['preprocessing']['df_path']

    if os.path.splitext(df_path)[1]:
        pass
    else:
        df_path = os.path.join(df_path,'FileFrame.csv')

    nifti_folder = os.path.join(os.path.dirname(DICOM_FOLDER),'NIFTI')
    print('applying monai transforms and splitting images')
    nifti_slices_folder = prep.preprocessImagesMonai(nifti_folder,x_image_size,y_image_size,z_image_size, df_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This is the preprocessing pipeline for a data or nifti folder \
                                                depening on what is specified in the config.yaml file.')
    parser.add_argument('-c', '--configFile', action='store', metavar='c', required=True,
                        help='pass here the config file path (from root or absolute) \
                            that should be used with your program')
    parser.add_argument('-m', '--mode', action='store', required=True,
                        help='set this flag to one of the folllwing values: \
                            \n "full" to run the whole preprocessing pipeline \
                            \n "sorting" to run only the dicom sorting \
                            \n "conversion" to run only dicom to nifti conversion \
                            \n "processing" to run only the nifti processing and slicing part')
    args = parser.parse_args()
    config = args.configFile
    mode = args.mode
    print(f'Preprocessing config file {config} using mode {mode}')
    if mode == "full":
        full_preprocessing(config)
    elif mode == "sorting":
        dicom_sorting(config)
    elif mode == "conversion":
        dicom_to_nifti_from_structured(config)
    elif mode == "processing":
        nifti_processing(config)
    else:
        print('invalid mode, see help for flag -m')

