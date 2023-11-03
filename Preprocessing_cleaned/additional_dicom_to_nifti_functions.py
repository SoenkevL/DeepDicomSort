import os
from glob import glob
import shutil
import dicom2nifti


def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def delete_directory(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
def convert_DICOM_to_NIFTI_dcm2niix(root_dir):
    # Convert all dicom files in the directory to nifti
    base_dir = os.path.dirname(os.path.normpath(root_dir))
    out_dir = os.path.join(base_dir, 'NIFTI')
    temp_dir = os.path.join(base_dir, 'TEMP')

    create_directory(out_dir)
    delete_directory(temp_dir)

    for root, dirs, files in os.walk(root_dir):
        # if len(files) > 0 and 'MR' in root:
        if len(files) > 0:
            create_directory(temp_dir)
            patient_ID = root.split(root_dir)[1]
            patient_ID = patient_ID.split(os.sep)[1]

            patient_out_folder = os.path.join(out_dir, patient_ID)
            create_directory(patient_out_folder)

            sub_directory_names = root.split(os.path.join(root_dir, patient_ID))[1]
            sub_directory_names = sub_directory_names.split(os.sep)[1:]

            nifti_file_name = '__'.join(sub_directory_names)

            system_command = 'dcm2niix' + ' -6 -z y -a y -b n -d 0 -e n -f ' + nifti_file_name + ' -g n -i n -l y -w 0 -o ' + temp_dir + ' ' + root
            os.system(system_command)

            # dcm2niix doesnt always use the specified file name, this is a work around
            nifti_out_files = glob(os.path.join(temp_dir, '*.nii.gz'))
            if len(nifti_out_files) > 0:
                shutil.move(nifti_out_files[0], os.path.join(patient_out_folder, nifti_file_name + '.nii.gz'))

            delete_directory(temp_dir)

    return out_dir

def convert_DICOM_to_NIFTI_dicom2nifti(root_dir):
    # Convert all dicom files in the directory to nifti
    base_dir = os.path.dirname(os.path.normpath(root_dir))
    out_dir = os.path.join(base_dir, 'NIFTI')
    temp_dir = os.path.join(base_dir, 'TEMP')

    create_directory(out_dir)
    delete_directory(temp_dir)

    for root, dirs, files in os.walk(root_dir):
        # if len(files) > 0 and 'MR' in root:
        if len(files) > 0:
            print(f'{len(files)} at {root}')
            patient_ID = root.split(root_dir)[1]
            patient_ID = patient_ID.split(os.sep)[1]

            patient_out_folder = os.path.join(out_dir, patient_ID)

            sub_directory_names = root.split(os.path.join(root_dir, patient_ID))[1]
            sub_directory_names = sub_directory_names.split(os.sep)[1:]

            nifti_file_name = '__'.join(sub_directory_names)
            patient_out_folder = os.path.join(patient_out_folder, nifti_file_name)
            create_directory(patient_out_folder)
            print(f'nifti filename: {nifti_file_name}')
            dicom2nifti.convert_directory(root, patient_out_folder)

    return out_dir