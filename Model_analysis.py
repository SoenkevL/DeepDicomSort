import json
import argparse
import pandas as pd
import os
import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, ConfusionMatrixDisplay, roc_auc_score, f1_score
import matplotlib.pyplot as plt

def initialize(outfile):
    print('initializing')
    out_file = outfile
    out_file_folder = out_file.split('/')[:-1]
    out_file_folder = str.join('/',out_file_folder)
    modelname = out_file.split('/')[-1].split(':')[0:-1]
    modelname = str.join(':',modelname)
    ResultFrame = pd.read_csv(out_file)
    meta_dict_file = out_file.replace('.csv','_metaDict.json')
    result_dict_file = out_file.replace('.csv','_resultDict.json')
    with open(meta_dict_file,'r') as mf:
        meta_dict = json.load(mf)
    return out_file_folder, modelname, ResultFrame, meta_dict

def splitImageID(dataframe):
    imageID = dataframe['imageID']
    sliceName = imageID.split('/')[-1].split('.nii.gz')[0]
    NIFTI_name = sliceName.split('__s')[0]
    slicenum = sliceName.split('__s')[1]
    NIFTI_path = imageID.split(NIFTI_name)[0]
    NIFTI_path = NIFTI_path.replace('NIFTI_SLICES','NIFTI')
    NIFTI_path = os.path.normpath(f'{NIFTI_path}{NIFTI_name}.nii.gz')
    return pd.Series([NIFTI_path, NIFTI_name, slicenum], index=['NIFTI_path', 'NIFTI_name', 'slicenum'])

def majorityVote(frame,N_classes):
    votes = np.zeros(N_classes)
    framelength = len(frame)
    for prediction in frame['prediction']:
        votes[prediction] += 1
    majorityClass = np.argmax(votes)
    certainty = np.max(votes)/framelength
    return pd.Series({'vote':majorityClass,'certainty':certainty})

def processDataframe(out_file_folder, modelname, ResultFrame, meta_dict):
    extraFrame = ResultFrame.apply(splitImageID, axis=1)
    ResultFrame = ResultFrame.merge(extraFrame, how='left', left_index=True, right_index=True, validate='one_to_one')
    NumSlicesPerClass=ResultFrame['slicenum'].nunique()
    ResultFrame = ResultFrame.set_index(['NIFTI_name'],drop=True)
    labelmap = meta_dict['labelmap']
    N_classes = len(labelmap)
    VotingFrame = ResultFrame[['prediction']].groupby('NIFTI_name').apply(majorityVote,N_classes)
    FullResultFrame = ResultFrame.merge(VotingFrame,how='left',on='NIFTI_name')
    FullResultFrame = FullResultFrame.set_index('slicenum',append=True).sort_index()
    FullResultFrame.to_csv(os.path.join(out_file_folder,f'{modelname}_ensamblePredictions.csv'),index=True)
    return FullResultFrame, NumSlicesPerClass

def createMetrics(FullResultFrame, out_file, NumSlicesPerClass, modelname, meta_dict, result_dict_file, certainties=[]):
    StringLabels = list(meta_dict['labelmap'].keys())
    NumericalLabels = list(meta_dict['labelmap'].values())
    resultDict = {}
    #results by slice basis
    mc = confusion_matrix(y_true=FullResultFrame['groundTruth'],y_pred=FullResultFrame['prediction'],labels=NumericalLabels)
    ac = balanced_accuracy_score(y_true=FullResultFrame['groundTruth'],y_pred=FullResultFrame['prediction'])
    f1 = f1_score(y_true=FullResultFrame['groundTruth'],y_pred=FullResultFrame['prediction'], average='macro')
    auc_score = roc_auc_score(y_true=FullResultFrame['groundTruth'],y_pred=FullResultFrame['prediction'], average='macro', multi_class='ovo')
    resultDict['ac_slice'] = ac
    resultDict['f1_slice'] = f1
    resultDict['auc_slice'] = auc_score
    mc_display = ConfusionMatrixDisplay(mc,display_labels=StringLabels)
    mc_display.plot(cmap='viridis')
    plt.grid(False)
    plt.xticks(rotation=90)
    plt.title(f'model {modelname} with ac {ac*100:.2f}%\n for indivdual slices')
    plt.savefig(os.path.join(os.path.split(out_file)[0],f'{modelname}_individualSlices_heatmap.png'))
    #majority vote without certainty
    mc = confusion_matrix(y_true=FullResultFrame['groundTruth'],y_pred=FullResultFrame['vote'],labels=NumericalLabels)
    mc = mc/NumSlicesPerClass
    ac = balanced_accuracy_score(y_true=FullResultFrame['groundTruth'],y_pred=FullResultFrame['vote'])
    f1 = f1_score(y_true=FullResultFrame['groundTruth'],y_pred=FullResultFrame['vote'], average='macro')
    auc_score = roc_auc_score(y_true=FullResultFrame['groundTruth'],y_pred=FullResultFrame['vote'], average='macro',
                              multi_class='ovo')
    resultDict['ac_vote_0'] = ac
    resultDict['f1_vote_0'] = f1
    resultDict['auc_vote_0'] = auc_score
    mc_display = ConfusionMatrixDisplay(mc,display_labels=StringLabels)
    mc_display.plot(cmap='viridis')
    plt.grid(False)
    plt.xticks(rotation=90)
    plt.title(f'model {modelname} with ac {ac*100:.2f}%\n for majorityVote')
    plt.savefig(os.path.join(os.path.split(out_file)[0],f'{modelname}_majorityVote_heatmap.png'))
    if certainties:
        c = certainties.split(',')
        certainties=[]
        for s in c:
            s = s.replace('[','')
            s = s.replace(']','')
            s = s.replace(' ','')
            try:
                s = float(s)
                certainties.append(s)
            except:
                print('something went wrong in putting in the certainties')
        for certaintyThreshhold in certainties:
            mc = confusion_matrix(y_true=FullResultFrame['groundTruth'][FullResultFrame['certainty']>=certaintyThreshhold],y_pred=FullResultFrame['vote'][FullResultFrame['certainty']>=certaintyThreshhold],labels=NumericalLabels)
            mc = mc/NumSlicesPerClass
            ytrue = FullResultFrame['groundTruth'][FullResultFrame['certainty']>=certaintyThreshhold]
            ypred = FullResultFrame['vote'][FullResultFrame['certainty']>=certaintyThreshhold]
            ac = balanced_accuracy_score(y_true=ytrue,y_pred=ypred)
            f1 = f1_score(y_true=ytrue,y_pred=ypred, average='macro')
            auc_score = roc_auc_score(y_true=ytrue,y_pred=ypred, average='macro',
                                      multi_class='ovo')
            resultDict[f'ac_vote_{certaintyThreshhold}'] = ac
            resultDict[f'f1_vote_{certaintyThreshhold}'] = f1
            resultDict[f'auc_vote_{certaintyThreshhold}'] = auc_score
            # mc_display = ConfusionMatrixDisplay(mc,display_labels=['T1','T1_c','T2','FLAIR','seg'])
            mc_display = ConfusionMatrixDisplay(mc,display_labels=StringLabels)
            mc_display.plot(cmap='viridis')
            plt.grid(False)
            plt.xticks(rotation=90)
            plt.title(f'model {modelname} with ac {ac*100:.2f}%\n for majorityVote with minimum certainty {certaintyThreshhold}')
            plt.savefig(os.path.join(os.path.split(out_file)[0],f'{modelname}_majorityVote_withCertaintyOf{certaintyThreshhold}_heatmap.png'))
    with open(result_dict_file,'w') as f:
        json.dump(resultDict,f)


def main(outfile, testing=False,certainties=[]):
    out_file_folder, modelname, ResultFrame_initial, meta_dict, result_dict_file = initialize(outfile)
    ResultFrame_processed, nslices = processDataframe(out_file_folder, modelname, ResultFrame_initial, meta_dict)
    if testing:
        createMetrics(
            ResultFrame_processed, outfile, NumSlicesPerClass=nslices, modelname=modelname,
            meta_dict=meta_dict, certainties=certainties, result_dict_file=result_dict_file
        )



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='This is the analysis of the results created in the model testing/prediction')
    parser.add_argument('-o','--outfile', action='store',metavar='o', help='pass here the filepath that was created by ModelTesting/predicting')
    parser.add_argument('-t','--testing',action='store_true', help='Pass this flag if testing should be done instead of only predictiong \n if testing results like accuracy and confusion matrices should be computed')
    parser.add_argument('--certainties',action='store',metavar='cert', default=[], help='specify a comma septerated string of certainties that should be used for the testing has to be in [0,1]. This needs to have the -t flag to be set')
    args = parser.parse_args()
    out_file = args.outfile
    testing = args.testing
    certainties = args.certainties
    main(out_file, testing, certainties)
    print('finished analysing results')

