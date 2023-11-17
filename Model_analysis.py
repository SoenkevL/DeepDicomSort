import json
import argparse
import pandas as pd
import os
import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, ConfusionMatrixDisplay, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import nibabel as nib

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
    return out_file_folder, modelname, ResultFrame, meta_dict, result_dict_file

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


def extractDataset(ID):
    if 'ADNI2' in ID:
        return 'ADNI2'
    elif 'ADNI3' in ID:
        return 'ADNI3'
    elif 'OASIS3' in ID:
        return 'OASIS3'
    elif 'egd' in ID:
        return 'egd'
    elif 'Rstudy' in ID:
        return 'rss'
    elif 'HeartBrain' in ID:
        return 'HeartBrain'
    elif 'small' in ID:
        return 'small'
    return None


def saveNifti(img_path, outfile):
    fig, ax = plt.subplots()
    nimg = nib.load(img_path)
    data = nimg.get_fdata()
    ax.imshow(data[:,:],cmap='gray')
    ax.axis('off')
    fig.savefig(outfile, pad_inches=0.01, bbox_inches='tight')
    plt.close('all')


def createMRIexamples(vis_frame_row, output_folder):
    outpath = os.path.join(output_folder, vis_frame_row['string_label'], vis_frame_row['dataset'])
    counter = 0
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    outfile = os.path.join(outpath, f"true_{vis_frame_row['string_label']}__pred_{vis_frame_row['string_vote']}"
                                    f"__orig_{vis_frame_row['dataset']}__{counter}.png")
    while os.path.exists(outfile):
        counter += 1
        outfile = os.path.join(outpath, f"true_{vis_frame_row['string_label']}__pred_{vis_frame_row['string_vote']}"
                                        f"__orig_{vis_frame_row['dataset']}__{counter}.png")
    saveNifti(vis_frame_row['imageID'], outfile)


def visualize_examples(out_file_folder, full_result_frame, meta_dict):
    label_name_list = list(meta_dict['labelmap'].keys())
    df = full_result_frame.copy()
    df['string_label'] = df['groundTruth'].apply(lambda x: label_name_list[x])
    df['dataset'] = df['imageID'].apply(extractDataset)
    labels = df['string_label'].unique()
    df['center_slide'] = df['imageID'].apply(lambda x: '__s12' in x)
    df['correctPrediction'] = df['vote'] == df['groundTruth']
    df['string_vote'] = df['vote'].apply(lambda x: label_name_list[int(x)])
    VisualizeFrame = df[df['center_slide']]
    from sklearn.model_selection import train_test_split
    visualize_frame = pd.DataFrame()
    for label in labels:
        temp1 = VisualizeFrame[VisualizeFrame['string_label']==label]
        datasets = temp1['dataset'].unique()
        for dataset in datasets:
            df = temp1[temp1['dataset']==dataset]
            try:
                visualize, _ = train_test_split(df, train_size=10, shuffle=True, random_state=42, stratify=df['correctPrediction'])
            except ValueError:
                visualize = df
            visualize_frame = pd.concat([visualize_frame,visualize], axis=0)
    if not os.path.exists(f'{out_file_folder}/example_images'):
        os.makedirs(f'{out_file_folder}/example_images')
    visualize_frame.to_csv(f'{out_file_folder}/example_images/vis_frame.csv')
    visualize_frame.apply(createMRIexamples, output_folder=f'{out_file_folder}/example_images', axis=1)



def createMetrics(FullResultFrame, out_file, NumSlicesPerClass, modelname, meta_dict, result_dict_file, certainties=[]):
    StringLabels = list(meta_dict['labelmap'].keys())
    NumericalLabels = list(meta_dict['labelmap'].values())
    resultDict = {}
    cols = FullResultFrame.columns
    raw_cols = [col for col in cols if 'raw' in col]
    #results by slice basis
    ac = balanced_accuracy_score(y_true=FullResultFrame['groundTruth'],y_pred=FullResultFrame['prediction'])
    f1 = f1_score(y_true=FullResultFrame['groundTruth'],y_pred=FullResultFrame['prediction'], average='macro')
    auc_score = roc_auc_score(y_true=FullResultFrame['groundTruth'],y_score=FullResultFrame[raw_cols], average='macro',
                              multi_class='ovo', labels=NumericalLabels
                              )
    resultDict['ac_slice'] = ac
    resultDict['f1_slice'] = f1
    resultDict['auc_slice'] = auc_score
    mc = ConfusionMatrixDisplay.from_predictions(y_true=FullResultFrame['groundTruth'],y_pred=FullResultFrame['prediction'],
                        labels=NumericalLabels, normalize='true', cmap='viridis', display_labels=StringLabels,
                        include_values=False)   
    plt.grid(False)
    plt.xticks(rotation=90)
    plt.title(f'model {modelname} with ac {ac*100:.2f}%\n for indivdual slices')
    plt.savefig(os.path.join(os.path.split(out_file)[0],f'{modelname}_individualSlices_heatmap.png'))
    #majority vote without certainty
    ac = balanced_accuracy_score(y_true=FullResultFrame['groundTruth'],y_pred=FullResultFrame['vote'])
    f1 = f1_score(y_true=FullResultFrame['groundTruth'],y_pred=FullResultFrame['vote'], average='macro')
    raw_preds = FullResultFrame[raw_cols].to_numpy()
    auc_score = roc_auc_score(y_true=FullResultFrame['groundTruth'],y_score=raw_preds, average='macro',
                              multi_class='ovo', labels=NumericalLabels
                              )
    resultDict['ac_vote_0'] = ac
    resultDict['f1_vote_0'] = f1
    resultDict['auc_vote_0'] = auc_score
    mc = ConfusionMatrixDisplay.from_predictions(y_true=FullResultFrame['groundTruth'], y_pred=FullResultFrame['vote'],
                        labels=NumericalLabels, normalize='true', cmap='viridis', display_labels=StringLabels,
                        include_values=False)
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
            mc = confusion_matrix(y_true=FullResultFrame['groundTruth'][FullResultFrame['certainty']>=certaintyThreshhold],
                                  y_pred=FullResultFrame['vote'][FullResultFrame['certainty']>=certaintyThreshhold],
                                  labels=NumericalLabels, normalize='true'
                                  )
            mc = mc/NumSlicesPerClass
            ytrue = FullResultFrame['groundTruth'][FullResultFrame['certainty']>=certaintyThreshhold]
            ypred = FullResultFrame['vote'][FullResultFrame['certainty']>=certaintyThreshhold]
            ac = balanced_accuracy_score(y_true=ytrue,y_pred=ypred)
            f1 = f1_score(y_true=ytrue,y_pred=ypred, average='macro')
            auc_score = roc_auc_score(y_true=ytrue,
                                      y_score=FullResultFrame[raw_cols][FullResultFrame['certainty']>=certaintyThreshhold],
                                      average='macro', multi_class='ovo', labels=NumericalLabels)
            resultDict[f'ac_vote_{certaintyThreshhold}'] = ac
            resultDict[f'f1_vote_{certaintyThreshhold}'] = f1
            resultDict[f'auc_vote_{certaintyThreshhold}'] = auc_score
            mc = ConfusionMatrixDisplay.from_predictions(y_true=ytrue, y_pred=ypred,
                        labels=NumericalLabels, normalize='true', cmap='viridis', display_labels=StringLabels,
                        include_values=False)
            plt.grid(False)
            plt.xticks(rotation=90)
            plt.title(f'model {modelname} with ac {ac*100:.2f}%\n for majorityVote with minimum certainty {certaintyThreshhold}')
            plt.savefig(os.path.join(os.path.split(out_file)[0],f'{modelname}_majorityVote_withCertaintyOf{certaintyThreshhold}_heatmap.png'))
    with open(result_dict_file,'w') as f:
        json.dump(resultDict,f, indent="")


def main(outfile, testing=False,certainties=[], vis=False):
    out_file_folder, modelname, ResultFrame_initial, meta_dict, result_dict_file = initialize(outfile)
    ResultFrame_processed, nslices = processDataframe(out_file_folder, modelname, ResultFrame_initial, meta_dict)
    if testing:
        createMetrics(
            ResultFrame_processed, outfile, NumSlicesPerClass=nslices, modelname=modelname,
            meta_dict=meta_dict, certainties=certainties, result_dict_file=result_dict_file
        )
    if vis:
        visualize_examples(out_file_folder, ResultFrame_processed, meta_dict)



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='This is the analysis of the results created in the model testing/prediction')
    parser.add_argument('-o','--outfile', action='store',metavar='o', help='pass here the filepath that was created by ModelTesting/predicting')
    parser.add_argument('-t','--testing',action='store_true', help='Pass this flag if testing should be done instead of only predictiong \n if testing results like accuracy and confusion matrices should be computed')
    parser.add_argument('-v', '--visualize', action='store_true', help='visualize examples of correclty and misspredicted classes')
    parser.add_argument('--certainties',action='store',metavar='cert', default=[], help='specify a comma septerated string of certainties that should be used for the testing has to be in [0,1]. This needs to have the -t flag to be set')
    args = parser.parse_args()
    out_file = args.outfile
    testing = args.testing
    certainties = args.certainties
    visualize = args.visualize
    main(out_file, testing, certainties, visualize)
    print('finished analysing results')

