import json
import argparse
import pandas as pd
import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

out_file = '.testing/acePredicitons/Predictions_DDS_model_epochs100_time_2023-09-28_08:28:09.239463/Predictions_DDS_model_epochs100_time_2023-09-28_08:28:09.239463.csv'

def initialize(outfile):
    out_file = outfile
    out_file_folder = out_file.split('/')[:-1]
    out_file_folder = str.join('/',out_file_folder)
    modelname = out_file.split('/')[-1].split(':')[0:-1]
    modelname = str.join(':',modelname)
    ResultFrame = pd.read_csv(out_file)
    meta_dict_file = out_file.replace('.csv','metaDict.json')
    with open(meta_dict_file,'r') as mf:
        label_dict = json.load(mf)
    return out_file_folder, modelname, ResultFrame, label_dict

def splitImageID1(imageID):
    sliceName = imageID.split('/')[-1].split('.nii.gz')[0]
    NIFTI_name = sliceName.split('__s')[0]
    slicenum = sliceName.split('__s')[1]
    return NIFTI_name

def splitImageID2(imageID):
    sliceName = imageID.split('/')[-1].split('.nii.gz')[0]
    NIFTI_name = sliceName.split('__s')[0]
    slicenum = sliceName.split('__s')[1]
    return slicenum

def majorityVote(frame,N_classes):
    votes = np.zeros(N_classes)
    framelength = len(frame)
    for prediction in frame['prediction']:
        votes[prediction] += 1
    majorityClass = np.argmax(votes)
    certainty = np.max(votes)/framelength
    return {'vote':majorityClass,'certainty':certainty}

def processDataframe(out_file_folder, modelname, ResultFrame):
    ResultFrame['NIFTI_name']= ResultFrame['imageID'].apply(splitImageID1)
    ResultFrame['slicenum'] = ResultFrame['imageID'].apply(splitImageID2)
    NumSlicesPerClass=ResultFrame['slicenum'].nunique()
    ResultFrame = ResultFrame.set_index(['NIFTI_name','slicenum'],drop=True)
    ResultFrame = ResultFrame.sort_index()
    VotingFrame = ResultFrame[['prediction']].groupby('NIFTI_name').apply(majorityVote,5)
    index = VotingFrame.index
    VotingFrame = pd.DataFrame(list(VotingFrame))
    VotingFrame['NIFTI_name'] = index
    FullResultFrame = pd.merge(ResultFrame,VotingFrame[['vote','certainty','NIFTI_name']],how='inner',on='NIFTI_name')
    FullResultFrame.to_csv(os.path.join(out_file_folder,f'{modelname}_ensamblePredictions.csv'),index=False)
    return FullResultFrame, NumSlicesPerClass

def createCF_matrix(FullResultFrame, NumSlicesPerClass, modelname, labelmap, certainties=[]):
    NumericalLabels = labelmap.keys() #important that the keys are the numbers and the values are the corresponding names
    StringLabels = labelmap.values()
    #results by slice basis
    mc = confusion_matrix(y_true=FullResultFrame['groundTruth'],y_pred=FullResultFrame['prediction'],labels=NumericalLabels)
    ac = accuracy_score(y_true=FullResultFrame['groundTruth'],y_pred=FullResultFrame['prediction'])
    mc_display = ConfusionMatrixDisplay(mc,display_labels=StringLabels)
    mc_display.plot(cmap='rocket')
    plt.grid(False)
    plt.title(f'model {modelname} with ac {ac*100:.2f}%\n for indivdual slices')
    plt.savefig(os.path.join(os.path.split(out_file)[0],f'{modelname}_individualSlices_heatmap.png'))
    #majority vote without certainty
    mc = confusion_matrix(y_true=FullResultFrame['groundTruth'],y_pred=FullResultFrame['vote'],labels=NumericalLabels)
    mc = mc/NumSlicesPerClass
    ac = accuracy_score(y_true=FullResultFrame['groundTruth'],y_pred=FullResultFrame['vote'])
    mc_display = ConfusionMatrixDisplay(mc,display_labels=StringLabels)
    mc_display.plot(cmap='rocket')
    plt.grid(False)
    plt.title(f'model {modelname} with ac {ac*100:.2f}%\n for majorityVote')
    plt.savefig(os.path.join(os.path.split(out_file)[0],f'{modelname}_majorityVote_heatmap.png'))
    if certainties:
        for certaintyThreshhold in certainties:
            mc = confusion_matrix(y_true=FullResultFrame['groundTruth'][FullResultFrame['certainty']>=certaintyThreshhold],y_pred=FullResultFrame['vote'][FullResultFrame['certainty']>=certaintyThreshhold],labels=[0,1,2,3,4])
            mc = mc/NumSlicesPerClass
            ac = accuracy_score(y_true=FullResultFrame['groundTruth'],y_pred=FullResultFrame['vote'])
            # mc_display = ConfusionMatrixDisplay(mc,display_labels=['T1','T1_c','T2','FLAIR','seg'])
            mc_display = ConfusionMatrixDisplay(mc,display_labels=['T1','T2','T2-FLAIR','PD','other'])
            mc_display.plot(cmap='rocket')
            plt.grid(False)
            plt.title(f'model {modelname} with ac {ac*100:.2f}%\n for majorityVote with minimum certainty {certaintyThreshhold}')
            plt.savefig(os.path.join(os.path.split(out_file)[0],f'{modelname}_majorityVote_withCertaintyOf{certaintyThreshhold}_heatmap.png'))

def main(outfile, testing=False,certainties=[]):
    out_file_folder, modelname, ResultFrame_initial, label_dict = initialize(outfile)
    ResultFrame_processed, nslices = processDataframe(out_file_folder, modelname, ResultFrame_initial)
    if testing:
        createCF_matrix(ResultFrame_processed, NumSlicesPerClass=nslices, modelname=modelname, labelmap=label_dict, certainties=certainties)



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='This is the analysis of the results created in the model testing/prediction')
    parser.add_argument('-o','--outfile', action='store',metavar='o', help='pass here the filepath that was created by ModelTesting')
    parser.add_argument('-t','--testing',action='store',metavar='t',default=False, help='Pass here if testing results like accuracy and confusion matrices should be computed')
    parser.add_argument('--certainties',action='store',metavar='cert', default=[], help='specify an array of certainties that should be used for the testing has to be in (0,1)')
    args = parser.parse_args()
    out_file = args.outfile
    testing = args.testing
    certainties = args.certainties
    main(out_file, testing, certainties)
    print('finished analysing results')

