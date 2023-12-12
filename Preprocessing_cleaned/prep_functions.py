import os
import shutil
from tqdm import tqdm
import numpy as np
import pandas as pd

import pydicom
from pydicom.errors import InvalidDicomError
import SimpleITK as sitk
import monai
from monai.data import FolderLayoutBase
from monai.config import PathLike
from monai.data.utils import create_file_basename
from monai.transforms import LoadImage, SaveImage, EnsureChannelFirst


def sort_DICOM_to_structured_folders(root_dir, df_path, move_files=False):
    '''
    functions to sort unstructured folders containing dicoms using dicom header info
    root_dir: root dir containing the dicom files
    df_path: path to the dataframe, has to end in filename.csv where filename is the name of the dataframe
    move_files: set to true if files should be moved out of the old folder instead of copied
    '''
    base_dir = os.path.dirname(os.path.normpath(root_dir))
    output_dir = os.path.join(base_dir, 'DICOM_STRUCTURED')
    dataPathList = []
    # To keep files seperate from following functions place in specific folder
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in tqdm(os.walk(root_dir)):
        # Check if there actually are any files in current folder
        if len(files) > 0:
            outputFolders = []
            dicom_output_folder = None
            for i_file_name in files:
                try:
                    # Try, so that only dicom files get moved (pydicom will give an error otherwise)
                    full_file_path = os.path.join(root, i_file_name)
                    dicom_data = pydicom.read_file(full_file_path, stop_before_pixels=True)

                    patient_ID = dicom_data.PatientID
                    study_date = dicom_data.StudyDate
                    scan_modality = dicom_data.Modality
                    study_instance_UID = dicom_data.StudyInstanceUID
                    series_instance_UID = dicom_data.SeriesInstanceUID

                    dicom_output_folder = os.path.join(output_dir, patient_ID,
                                                       study_date, scan_modality,
                                                       study_instance_UID,
                                                       series_instance_UID)

                    if not os.path.exists(dicom_output_folder):
                        os.makedirs(dicom_output_folder)
                    if move_files:
                        shutil.move(full_file_path, dicom_output_folder)
                    else:
                        try: 
                            shutil.copy(full_file_path, dicom_output_folder)
                            outputFolders.append(dicom_output_folder)
                        except shutil.SameFileError:
                            outputFolders.append(dicom_output_folder)
                        except: 
                            pass
                except:
                    pass
            for struct_folder in set(outputFolders):
                dataPathList.append([root, struct_folder])
    df = pd.DataFrame(dataPathList, columns=['originPath','structuredDicomPath'])
    df.to_csv(df_path, index=False)
    return output_dir

def split_in_series(root_dir, df_path):
    '''
    This function deals with multiple files beeing part of the same series in one dicom folder
    root_dir: structured dicom dir produced by sort_DICOM_to_structured_folders
    df_path: path to the dataframe, has to end in filename.csv where filename is the name of the dataframe
    '''
    splitedSeries = []
    for root, dirs, files in tqdm(os.walk(root_dir)):
        if len(files) > 0:
            hash_list = list()
            for i_file in files:
                i_dicom_file = os.path.join(root, i_file)
                try:
                    temp_dicom = pydicom.read_file(i_dicom_file, stop_before_pixels=True)
                except InvalidDicomError:
                    continue

                # Fields to split on
                if (0x28, 0x10) in temp_dicom:
                    N_rows = temp_dicom[0x28, 0x10].value
                else:
                    N_rows = -1
                if (0x28, 0x11) in temp_dicom:
                    N_columns = temp_dicom[0x28, 0x11].value
                else:
                    N_columns = -1

                # some very small deviations can be expected, so we round
                if (0x28, 0x30) in temp_dicom:
                    pixel_spacing = temp_dicom[0x28, 0x30].value
                    if not isinstance(pixel_spacing, str):
                        pixel_spacing = np.round(pixel_spacing, decimals=6)
                    else:
                        pixel_spacing = [-1, -1]
                else:
                    pixel_spacing = [-1, -1]
                if 'RepetitionTime' in temp_dicom:
                    try:
                        repetition_time = float(temp_dicom[0x18, 0x80].value)
                        repetition_time = np.round(repetition_time, decimals=6)
                    except:
                        repetition_time = -1
                else:
                    repetition_time = -1
                try:
                    echo_time = np.round(temp_dicom[0x18, 0x81].value, decimals=6)
                except:
                    echo_time = -1

                diff_tuple = (N_rows, N_columns, pixel_spacing[0], pixel_spacing[1],
                              repetition_time, echo_time)
                hash_tuple = hash(diff_tuple)

                hash_list.append(hash_tuple)

            if len(set(hash_list)) > 1:
                N_sets = len(set(hash_list))

                i_sets = range(1, N_sets + 1)
                upper_folder = os.path.dirname(root)
                scan_name = os.path.basename(os.path.normpath(root))
                for i_set in i_sets:
                    new_scan_dir = os.path.join(upper_folder, scan_name + '_Scan_' + str(i_set))
                    splitedSeries.append((root, new_scan_dir))
                    if not os.path.exists(new_scan_dir):
                        os.makedirs(new_scan_dir)

                _, reverse_indices = np.unique(hash_list, return_inverse=True)

                for i_dicom, index_type in zip(files, reverse_indices):
                    new_scan_dir = os.path.join(upper_folder, scan_name + '_Scan_' + str(i_sets[index_type]))
                    try:
                        shutil.move(os.path.join(root, i_dicom), new_scan_dir)
                    except shutil.Error:
                        print(f'split in series: destination {new_scan_dir} allready exists')
                    except:
                        pass
                shutil.rmtree(root)
    if splitedSeries:
        df = pd.read_csv(df_path)
        df = df[['originPath', 'structuredDicomPath']]
        priorRoot = None
        for root, newScanDir in splitedSeries:
            if root != priorRoot:
                priorRoot = root
                datapath = df['originPath'][df['structuredDicomPath']==root]
                DropIndex = df[df['structuredDicomPath']==root].index
                df.drop(index=DropIndex, inplace=True)
            df = pd.concat((df,pd.DataFrame({'originPath':datapath, 'structuredDicomPath':newScanDir})), axis=0)
            df.reset_index()
        df.to_csv(df_path, index=False)

class CustomFolderLayout(FolderLayoutBase):

    def __init__(
            self,
            output_dir: PathLike,
            postfix: str = "",
            extension: str = "",
            parent: bool = False,
            makedirs: bool = False,
            data_root_dir: PathLike = "",
    ):
        self.output_dir = output_dir
        self.postfix = postfix
        self.ext = extension
        self.parent = parent
        self.makedirs = makedirs
        self.data_root_dir = data_root_dir


    def filename(self, subject: PathLike = "subject", idx=None, **kwargs) -> PathLike:
        if not '.dcm' in subject or '.nii.gz' in subject:
            subject = subject + '.placeholder'
        full_name = create_file_basename(
            postfix=self.postfix,
            input_file_name=subject,
            folder_path=self.output_dir,
            data_root_dir=self.data_root_dir,
            separate_folder=self.parent,
            patch_index=idx,
            makedirs=self.makedirs,
        )
        for k, v in kwargs.items():
            full_name += f"_{k}-{v}"
        if self.ext is not None:
            ext = f"{self.ext}"
            full_name += f".{ext}" if ext and not ext.startswith(".") else f"{ext}"
        return full_name

def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def delete_directory(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)

def convert_DICOM_to_NIFTI_monai(root_dir, df_path):
    '''
    convert all dicom files in root_dir to nifti, will be saved in a new folder called NIFTI
    root_dir: directory to convert
    df_path: path to the csv file for the dataframe creation which logs the process
    '''
    base_dir = os.path.dirname(os.path.normpath(root_dir))
    out_dir = os.path.join(base_dir, 'NIFTI')

    create_directory(out_dir)
    direcList = []
    for root, dirs, files in tqdm(os.walk(root_dir)):
        # if len(files) > 0 and 'MR' in root:
        if len(files) > 0:
            try:
                image = LoadImage(image_only=False)(root)
                Timage = EnsureChannelFirst()(image[0], image[1])
                SaveImage(folder_layout=CustomFolderLayout(output_dir=out_dir,extension='.nii.gz',data_root_dir=root_dir, makedirs=True),squeeze_end_dims=True)(Timage, image[1])
                temp = CustomFolderLayout(output_dir=out_dir,extension='.nii.gz',data_root_dir=root_dir, makedirs=True)
                newName = temp.filename(root)
                direcList.append([root, newName])
            except:
                print(f'failed to convert dicomFolder: {root}')
                direcList.append(['root',None])
                continue
    temp = pd.DataFrame(direcList, columns=['structuredDicomPath', 'NIFTI_path'])
    df = pd.read_csv(df_path)
    try:
        df = df.merge(temp, how='left', on='structuredDicomPath')
    except:
        df_path = os.path.join(os.path.dirname(df_path), 'Dicom_to_Nifti_frame.csv')
        df = temp
    df.to_csv(df_path, index=False)
    return out_dir

def extract_4D_images(root_dir):
    '''
    extract the number of channels of a nifti image, if 4 the first one will be kept and the filepath
    will be added to a txt which is returned by the function
    '''
    base_dir = os.path.dirname(os.path.normpath(root_dir))
    data_dir = os.path.join(base_dir, 'DATA')
    create_directory(data_dir)
    out_4D_file = os.path.join(data_dir, 'Scans_4D.txt')

    with open(out_4D_file, 'w') as the_file:
        for root, dirs, files in tqdm(os.walk(root_dir)):
            for i_file in files:
                if '.nii.gz' in i_file:
                    image_file = os.path.join(root, i_file)

                    image = sitk.ReadImage(image_file, sitk.sitkFloat32)
                    if image.GetDimension() == 4:
                        file_name = i_file.split('.nii.gz')[0]
                        image_size = list(image.GetSize())
                        image_size[3] = 0
                        image = sitk.Extract(image,
                                             size=image_size,
                                             index=[0, 0, 0, 0])

                        sitk.WriteImage(image, image_file)

                        the_file.write(file_name + '\n')
    return out_4D_file

def create_label_file(nifti_dir,base_dir, images_4D_file, name='Labels.txt'):
    '''
    creates the raw label file with 0 as the the default class label and extra info from the 4d step
    '''
    base_dir = base_dir
    data_dir = os.path.join(base_dir, 'DATA')
    label_file = os.path.join(data_dir, name)

    images_4D = np.genfromtxt(images_4D_file, dtype='str')

    with open(label_file, 'w') as the_file:
        for root, dirs, files in os.walk(nifti_dir):
            for i_file in files:
                if '.nii.gz' in i_file:
                    file_name = i_file.split('.nii.gz')[0].split('_')[0:-1]
                    file_name = '_'.join(file_name)
                    if file_name in images_4D:
                        is_4D = '1'
                    else:
                        is_4D = '0'

                    file_location = os.path.join(root, i_file)

                    out_elements = [file_location, '0', is_4D]

                    the_file.write('\t'.join(out_elements) + '\n')

    return label_file


def preprocessImagesMonai(niftiDirec, x, y, z, df_path):
    '''
    function which preprocesses the nifti volumes and saves them afterwards to individual slices
    resamples the images to x*y*z, puts them in LAS orientation, scales volume intensity between 0 and 1
    and saves 25 individual slices per volume. Logs in df_path csv if conversion was successfull
    '''
    root_dataFolder = os.path.split(niftiDirec)[0]

    NIFTI_direc = niftiDirec
    nifti_slices_direc = os.path.join(root_dataFolder,'NIFTI_SLICES')
    if not os.path.exists(NIFTI_direc):
        print('no nifti files found')
        return None
    if not os.path.exists(nifti_slices_direc):
        os.mkdir(nifti_slices_direc)

    print('>>> extracting 4d info')
    images_4D_file = extract_4D_images(NIFTI_direc) #could be multiprocessed

    filepaths = []
    filenames = []
    for root, dirs, files in os.walk(NIFTI_direc):
        for i_file in files:
            if '.nii.gz' in i_file:
                filepaths.append(os.path.join(root,i_file))
    print(len(filepaths))
    dataset = [{'image':filepath,'label':filepath} for filepath in filepaths]


    class SaveIndividualSlices(monai.transforms.MapTransform):
        def __init__(self, keys, targetDir, split):
            self.keys = keys
            self.targetDir = targetDir
            if split:
                self.outdir = os.path.join(self.targetDir,split)
            else:
                self.outdir = self.targetDir

        def __call__(self, data):
            converted_Files = []
            os.makedirs(self.targetDir,exist_ok=True)
            depth = data[self.keys[0]].shape[-1]
            converted = True
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
                    print(f'runtime error when trying to save image {data[self.keys[1]]} to slice{i}')
                    converted = False
                    break
            if converted:
                converted_Files.append(data[self.keys[1]])
            return data, data[self.keys[1]]

    dataTransform = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=['image'], image_only=False, ensure_channel_first=False),
            monai.transforms.EnsureChannelFirstd(keys=['image']),
            monai.transforms.Orientationd(keys=['image'],axcodes='LAS'),
            monai.transforms.Resized(keys=['image'], spatial_size=[x, y, z]),
            monai.transforms.ScaleIntensityd(keys=['image']),
            SaveIndividualSlices(keys=['image', 'label'], targetDir=nifti_slices_direc,split=None)
        ],
        lazy=True
    )

    ds = monai.data.Dataset(dataset, dataTransform)
    dl = monai.data.DataLoader(ds, batch_size=1, num_workers=8)
    dl_iter = iter(dl)
    print('>>> create data set')
    All_converted_files = []
    for _ in tqdm(range(len(ds))):
        try:
            _, converted_Files = next(dl_iter)
            [All_converted_files.append(file) for file in converted_Files]
        except StopIteration:
            break
        finally:
            continue
    print(converted_Files)

    create_label_file(nifti_slices_direc,root_dataFolder, images_4D_file,'Labels.txt')

    df = pd.read_csv(df_path)
    try:
        for niftipath in All_converted_files:
            df.loc[df['NIFTI_path']==niftipath,'sliced'] = True
    except:
        df_path = os.path.join(os.path.dirname(df_path),'ConversionFrame.csv')
        df['NIFTI_path'] = pd.Series(All_converted_files)
        df['sliced'] = True
    df.to_csv(df_path, index=False)

    return nifti_slices_direc


def protectConfig(configFile):
    '''
    creates a copy of the config file in order to prevent overwritting it in an undesired way due to an error in the
    code
    '''
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
                'Institution':None,
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
            try:
                data = pydicom.read_file(os.path.join(root, files[0]), stop_before_pixels=True)
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
                    dicomAttributeDict['Institution'] = data.InstitutionName
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
            except:
                continue
    df_dicom = pd.DataFrame(dicts)
    df_dicom.to_csv(outPath, index=False)
    return outPath

