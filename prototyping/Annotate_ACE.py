import pandas as pd
def unify_mapping_names(name):
    name = name.strip()
    return name.replace(' ', '_')

def extract_scantype_name_from_datapath(datapath):
    datapath = datapath.split('/')[9]
    datapath = datapath.strip()
    datapath = '-'.join(datapath.split('-')[1:])
    return datapath

def assign_label(extracted_scantype, xnat_frame_cleaned):
    cols = list(xnat_frame_cleaned.columns)
    for col in cols:
        if extracted_scantype in list(xnat_frame_cleaned.loc[:,col]):
            return col
    return None

if __name__ == '__main__':
    FileFrame = pd.read_csv('/data/scratch/r098375/data/ACE_new/FileFrame.csv')
    Results = pd.read_csv('/data/scratch/r098375/data/ACE/post_processing/Predictions_DDS_model_epochs20_time_2023-11-19_22:56_ensamblePredictions.csv')
    XNATFrame = pd.read_csv('/trinity/home/r098375/DDS/DeepDicomSort/.finalExperiments/exp4/scantype_mapping.csv', header=0, delimiter=';')

    Results_depr = Results[Results['slicenum']==10][['vote','NIFTI_path']]

    CombinedFrame = Results_depr.merge(FileFrame, how='inner', left_on='NIFTI_path', right_on='NIFTI_path')
    CombinedFrame = CombinedFrame.drop('sliced', axis=1)
    CombinedFrame['extracted_scantype'] = CombinedFrame['originPath'].apply(extract_scantype_name_from_datapath)

    XNAT_Frame_cleanedNames = XNATFrame.apply(unify_mapping_names)

    CombinedFrame['cleaned_label'] = CombinedFrame['extracted_scantype'].apply(assign_label, xnat_frame_cleaned=XNAT_Frame_cleanedNames)
    CombinedFrame.to_csv('Annotated_File_Frame.csv', index=False)