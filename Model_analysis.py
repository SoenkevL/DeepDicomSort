import json
import argparse
import pandas as pd
import os
import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score
import matplotlib.pyplot as plt
import nibabel as nib

def initialize(outfile):
    '''
    uses the outfile path in order to infere the name of the meta_dictionary and use these to load the outputfolder,
    modelname, loads the result frame and meta_dict
    inputs:
    -outfile: filepath of the resultfile that is to be analyzed
    outputs:
    -out_file_folder: folder where the result file is
    -modelname: name of the model which was used for predicting
    -ResultFrame: the loaded result frome from the out_file
    -meta_dict: the loaded dictionary from the meta_dict_file
    -result_dict_file: the file where the results will be stored
    '''
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
    '''
    uses the imageID in order to find the NIFTI_path, slicenumber and NIFTI name.
    This is supposed to be used using pandas apply on rows of a datamframe
    input:
    -dataframe: dataframe to calculate from. Needs to have 'imageID' as column
    outputs:
    -pandas_Series: contains the cols [NIFTI_path, NIFTI_name, slicenumber]
    '''
    imageID = dataframe['imageID']
    sliceName = imageID.split('/')[-1].split('.nii.gz')[0]
    NIFTI_name = sliceName.split('__s')[0]
    slicenum = sliceName.split('__s')[1]
    NIFTI_path = imageID.split(NIFTI_name)[0]
    NIFTI_path = NIFTI_path.replace('NIFTI_SLICES','NIFTI')
    NIFTI_path = os.path.normpath(f'{NIFTI_path}{NIFTI_name}.nii.gz')
    return pd.Series([NIFTI_path, NIFTI_name, slicenum], index=['NIFTI_path', 'NIFTI_name', 'slicenum'])

def majorityVote(frame,N_classes):
    '''
    calculates the majority vote off predictions in a dataframe assuming there are N_different classes to predict
    inputs:
    -frame: dataframe to caclulate predictions from, needs to have 'prediction' column
    -N_classes: integer of how many classes there are in total to predict
    outputs:
    -pandas_Series: contains columns ['vote', 'certainty'] with same row length as the input dataframe
    '''
    votes = np.zeros(N_classes)
    framelength = len(frame)
    for prediction in frame['prediction']:
        votes[prediction] += 1
    majorityClass = np.argmax(votes)
    certainty = np.max(votes)/framelength
    return pd.Series({'vote':majorityClass,'certainty':certainty})

def processDataframe(out_file_folder, modelname, ResultFrame, meta_dict):
    '''
    Uses the functions splitImageID and majorityVote in order to create the ensemble frame from the predictions frame.
    inputs:
    -out_file_folder: folder where the original ouput file is stored
    -modelname: name of the model used for the predictions
    -ResultFrame: Frame containing the predictions of the model
    -meta_dict: dictionary containing informations about the training and testing process aswell as the classes
    outputs:
    FullResultFrame: original Result frame with the extra columns from splitImageID and majorityVote
    NumSlicesPerClass: returns how many slices per scan were used in the dataframe
    '''
    #extract the name, path and slicenum and add them to the original frame
    extraFrame = ResultFrame.apply(splitImageID, axis=1)
    ResultFrame = ResultFrame.merge(extraFrame, how='left', left_index=True, right_index=True, validate='one_to_one')
    #extract how many slices are used per scan
    NumSlicesPerClass=ResultFrame['slicenum'].nunique()
    ResultFrame = ResultFrame.set_index(['NIFTI_name'],drop=True)
    #load the labelmap to figure out how many classes there are
    labelmap = meta_dict['labelmap']
    N_classes = len(labelmap)
    #for each NIFTI_name (means one scan) it applies the majority vote
    VotingFrame = ResultFrame[['prediction']].groupby('NIFTI_name').apply(majorityVote,N_classes)
    FullResultFrame = ResultFrame.merge(VotingFrame,how='left',on='NIFTI_name')
    FullResultFrame = FullResultFrame.set_index('slicenum',append=True).sort_index()
    FullResultFrame.to_csv(os.path.join(out_file_folder,f'{modelname}_ensamblePredictions.csv'),index=True)
    return FullResultFrame, NumSlicesPerClass


def extractDataset(ID):
    '''
    extracts the dataset from the ID, is hard coded and needs to be adapted to the datasets used
    inputs:
    -ID: imageID from the dataframe
    outputs:
    -dataset: returns a name from the list of it is part of the ID
    '''
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
    elif 'parel' in ID:
        return 'parelsnoer'
    elif 'ACE' in ID:
        return 'ace'
    return None


def saveNifti(img_path, outfile):
    '''
    saves a nifti image
    inputs:
    -img_path: path where the nifti is stored
    -outfile: path where the image is stored
    '''
    fig, ax = plt.subplots()
    nimg = nib.load(img_path)
    data = nimg.get_fdata()
    ax.imshow(data[:,:],cmap='gray')
    ax.axis('off')
    fig.savefig(outfile, pad_inches=0.01, bbox_inches='tight')
    plt.close('all')


def createMRIexamples(vis_frame_row, output_folder):
    '''
    uses rows from a dataframe in order to create an example image with saveNifti.
    Should be using pandas apply along the row axis of a dataframe.
    inputs:
    -vis_frame_row: row from the dataframe that is to be visualized
    -output_folder: folder where images are stored
    '''
    outpath = os.path.join(output_folder, vis_frame_row['string_label'], vis_frame_row['dataset'])
    counter = 0
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    outfile = os.path.join(outpath, f"true_{vis_frame_row['string_label']}__pred_{vis_frame_row['string_vote']}"
                                    f"__cert_{vis_frame_row['certainty']}"
                                    f"__orig_{vis_frame_row['dataset']}__{counter}.png")
    while os.path.exists(outfile):
        counter += 1
        outfile = os.path.join(outpath, f"true_{vis_frame_row['string_label']}__pred_{vis_frame_row['string_vote']}"
                                        f"__cert_{vis_frame_row['certainty']}"
                                        f"__orig_{vis_frame_row['dataset']}__{counter}.png")
    saveNifti(vis_frame_row['imageID'], outfile)


def visualize_examples(out_file_folder, full_result_frame, meta_dict):
    '''
    creates a datafrane from the full result frame in order to visualize certain examples from it. These are picked
    in order to represent different predictions and misspredictions of the different classes in a strategic manner.
    inputs:
    -out_file_folder: folder where the images and visualize frame are stored
    -full_result_frame: the ensemble frame which allready needs to have vote and certainty cols
    -meta_dict: meta dictionary with info about the training and testing process (used for labels)
    '''
    #create a list of the labels
    label_name_list = list(meta_dict['labelmap'].keys())
    df = full_result_frame.copy()
    #create string labels for the classes and extract which dataset they belong to
    df['string_label'] = df['groundTruth'].apply(lambda x: label_name_list[x])
    df['dataset'] = df['imageID'].apply(extractDataset)
    labels = df['string_label'].unique()
    #extract the center slide for each image (slide 12 in this case)
    df['center_slide'] = df['imageID'].apply(lambda x: '__s12' in x)
    #add column that checks if the prediction is true or not
    df['correctPrediction'] = df['vote'] == df['groundTruth']
    df['string_vote'] = df['vote'].apply(lambda x: label_name_list[int(x)])
    #only keep the center slides for all scans
    VisualizeFrame = df[df['center_slide']]
    from sklearn.model_selection import train_test_split
    visualize_frame = pd.DataFrame()
    #go through all labels
    for label in labels:
        temp1 = VisualizeFrame[VisualizeFrame['string_label']==label]
        datasets = temp1['dataset'].unique()
        #for each dataset in a label
        for dataset in datasets:
            df = temp1[temp1['dataset']==dataset]
            #create a split of the images of size 10 which represent correctly and incorrectly classifed labels
            try:
                visualize, _ = train_test_split(df, train_size=10, shuffle=True, random_state=42, stratify=df['correctPrediction'])
            except ValueError:
                visualize = df
            visualize_frame = pd.concat([visualize_frame,visualize], axis=0)
    if not os.path.exists(f'{out_file_folder}/example_images'):
        os.makedirs(f'{out_file_folder}/example_images')
    #save the frame of the images that where visualized
    visualize_frame.to_csv(f'{out_file_folder}/example_images/vis_frame.csv')
    #to each row apply the createMRIexamples function to save an image with descriptive file path
    visualize_frame.apply(createMRIexamples, output_folder=f'{out_file_folder}/example_images', axis=1)



def createMetrics(FullResultFrame, out_file, NumSlicesPerClass, modelname, meta_dict, result_dict_file, certainties=[]):
    '''
    Creates metrics for the enesemble Frame. The metrics are calculated on a per slice basis as well as per vote basis.
    Additonally using the certainties different thresholds can be set to caclulate metrics for.
    The calculated metrics are ac, balanced ac, f1*, prec*, recall*, roc_auc_score, weighted roc_auc_score
    The metrics with * are calculated using micro and macro weight
    All metrics and figures will be saved to additonal files in the original result file folder
    inputs:
    -FullResultFrame: the ensemble Frame containg vote, certainties etc
    -out_file: where the original results were stored to infere where the figs are saved
    -NumSlicesPerClass: how many slices per class are used
    -modelname: Name of the model that was used for predicting
    -meta_dict: meta dictionary to infere the label names
    -result_dict_file: file where the result dictionary is stored
    -certainties: the certainties for which to calc metrics
    '''
    StringLabels = list(meta_dict['labelmap'].keys())
    NumericalLabels = list(meta_dict['labelmap'].values())
    resultDict = {}
    cols = FullResultFrame.columns
    raw_cols = [col for col in cols if 'raw' in col]
    #results by slice basis
    ac = balanced_accuracy_score(y_true=FullResultFrame['groundTruth'],y_pred=FullResultFrame['prediction'])
    f1 = f1_score(y_true=FullResultFrame['groundTruth'],y_pred=FullResultFrame['prediction'], average='macro')
    prec = precision_score(y_true=FullResultFrame['groundTruth'],y_pred=FullResultFrame['prediction'], average='macro')
    recall = recall_score(y_true=FullResultFrame['groundTruth'],y_pred=FullResultFrame['prediction'], average='macro')
    auc_score = roc_auc_score(y_true=FullResultFrame['groundTruth'],y_score=FullResultFrame[raw_cols], average='macro',
                              multi_class='ovo', labels=NumericalLabels
                              )
    resultDict['ac_slice__balanced'] = ac
    resultDict['f1_slice__macro'] = f1
    resultDict['prec_slice__macro'] = prec
    resultDict['recall_slice__macro'] = recall
    resultDict['auc_slice__macro'] = auc_score
    ac = accuracy_score(y_true=FullResultFrame['groundTruth'], y_pred=FullResultFrame['prediction'])
    f1 = f1_score(y_true=FullResultFrame['groundTruth'], y_pred=FullResultFrame['prediction'], average='micro')
    prec = precision_score(y_true=FullResultFrame['groundTruth'], y_pred=FullResultFrame['prediction'], average='micro')
    recall = recall_score(y_true=FullResultFrame['groundTruth'], y_pred=FullResultFrame['prediction'], average='micro')
    auc_score = roc_auc_score(y_true=FullResultFrame['groundTruth'], y_score=FullResultFrame[raw_cols],
                              average='weighted',
                              multi_class='ovo', labels=NumericalLabels
                              )
    resultDict['ac_slice__'] = ac
    resultDict['f1_slice__micro'] = f1
    resultDict['prec_slice__micro'] = prec
    resultDict['recall_slice__micro'] = recall
    resultDict['auc_slice__weighted'] = auc_score
    mc = ConfusionMatrixDisplay.from_predictions(y_true=FullResultFrame['groundTruth'],y_pred=FullResultFrame['prediction'],
                        labels=NumericalLabels, normalize='true', cmap='viridis', display_labels=StringLabels,
                        include_values=False)   
    plt.grid(False)
    plt.xticks(rotation=45)
    plt.tight_layout(pad=0.05)
    plt.savefig(os.path.join(os.path.split(out_file)[0],f'{modelname}_individualSlices_heatmap.png'))
    #majority vote without certainty
    ac = balanced_accuracy_score(y_true=FullResultFrame['groundTruth'],y_pred=FullResultFrame['vote'])
    f1 = f1_score(y_true=FullResultFrame['groundTruth'],y_pred=FullResultFrame['vote'], average='macro')
    prec = precision_score(y_true=FullResultFrame['groundTruth'],y_pred=FullResultFrame['vote'], average='macro')
    recall = recall_score(y_true=FullResultFrame['groundTruth'],y_pred=FullResultFrame['vote'], average='macro')
    raw_preds = FullResultFrame[raw_cols].to_numpy()
    auc_score = roc_auc_score(y_true=FullResultFrame['groundTruth'],y_score=raw_preds, average='macro',
                              multi_class='ovo', labels=NumericalLabels
                              )
    resultDict['ac_vote_0__balanced'] = ac
    resultDict['f1_vote_0__macro'] = f1
    resultDict['prec_vote_0__macro'] = prec
    resultDict['recall_vote_0__macro'] = recall
    resultDict['auc_vote_0__macro'] = auc_score
    ac = accuracy_score(y_true=FullResultFrame['groundTruth'], y_pred=FullResultFrame['vote'])
    f1 = f1_score(y_true=FullResultFrame['groundTruth'], y_pred=FullResultFrame['vote'], average='micro')
    prec = precision_score(y_true=FullResultFrame['groundTruth'], y_pred=FullResultFrame['vote'], average='micro')
    recall = recall_score(y_true=FullResultFrame['groundTruth'], y_pred=FullResultFrame['vote'], average='micro')
    raw_preds = FullResultFrame[raw_cols].to_numpy()
    auc_score = roc_auc_score(y_true=FullResultFrame['groundTruth'], y_score=raw_preds, average='weighted',
                              multi_class='ovo', labels=NumericalLabels
                              )
    resultDict['ac_vote_0__'] = ac
    resultDict['f1_vote_0__micro'] = f1
    resultDict['prec_vote_0__micro'] = prec
    resultDict['recall_vote_0__micro'] = recall
    resultDict['auc_vote_0__weighted'] = auc_score
    mc = ConfusionMatrixDisplay.from_predictions(y_true=FullResultFrame['groundTruth'], y_pred=FullResultFrame['vote'],
                        labels=NumericalLabels, normalize='true', cmap='viridis', display_labels=StringLabels,
                        include_values=False)
    plt.grid(False)
    plt.xticks(rotation=45)
    plt.tight_layout(pad=0.05)
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
            ytrue = FullResultFrame['groundTruth'][FullResultFrame['certainty']>=certaintyThreshhold]
            ypred = FullResultFrame['vote'][FullResultFrame['certainty']>=certaintyThreshhold]
            ac = balanced_accuracy_score(y_true=ytrue,y_pred=ypred)
            f1 = f1_score(y_true=ytrue,y_pred=ypred, average='macro')
            prec = precision_score(y_true=ytrue,y_pred=ypred, average='macro')
            recall = recall_score(y_true=ytrue,y_pred=ypred, average='macro')
            auc_score = roc_auc_score(y_true=ytrue,
                                      y_score=FullResultFrame[raw_cols][FullResultFrame['certainty']>=certaintyThreshhold],
                                      average='macro', multi_class='ovo', labels=NumericalLabels)
            resultDict[f'ac_vote_{certaintyThreshhold}__balanced'] = ac
            resultDict[f'f1_vote_{certaintyThreshhold}__macro'] = f1
            resultDict[f'prec_vote_{certaintyThreshhold}__macro'] = prec
            resultDict[f'recall_vote_{certaintyThreshhold}__macro'] = recall
            resultDict[f'auc_vote_{certaintyThreshhold}__macro'] = auc_score
            ac = accuracy_score(y_true=ytrue, y_pred=ypred)
            f1 = f1_score(y_true=ytrue, y_pred=ypred, average='micro')
            prec = precision_score(y_true=ytrue, y_pred=ypred, average='micro')
            recall = recall_score(y_true=ytrue, y_pred=ypred, average='micro')
            auc_score = roc_auc_score(y_true=ytrue,
                                    y_score=FullResultFrame[raw_cols][
                                        FullResultFrame['certainty'] >= certaintyThreshhold],
                                    average='weighted', multi_class='ovo', labels=NumericalLabels)
            resultDict[f'ac_vote_{certaintyThreshhold}__'] = ac
            resultDict[f'f1_vote_{certaintyThreshhold}__micro'] = f1
            resultDict[f'prec_vote_{certaintyThreshhold}__micro'] = prec
            resultDict[f'recall_vote_{certaintyThreshhold}__micro'] = recall
            resultDict[f'auc_vote_{certaintyThreshhold}__micro'] = auc_score
            mc = ConfusionMatrixDisplay.from_predictions(y_true=ytrue, y_pred=ypred,
                        labels=NumericalLabels, normalize='true', cmap='viridis', display_labels=StringLabels,
                        include_values=False)
            plt.grid(False)
            plt.xticks(rotation=45)
            plt.tight_layout(pad=0.05)
            plt.savefig(os.path.join(os.path.split(out_file)[0],f'{modelname}_majorityVote_withCertaintyOf{certaintyThreshhold}_heatmap.png'))
    with open(result_dict_file,'w') as f:
        json.dump(resultDict,f, indent="")


def main(outfile, testing=False,certainties=[], vis=False):
    '''
    main function of the analysis
    inputs:
    -outfile: the file containing the predicitons from the model
    -testing: this calculates metrics, needs to have groundtruth labels in order to be activated
    -certainties: certainties for metrics calculations, see createMetrics
    -vis: If examples should be visualized (currently also needs ground truth but could be adapted)
    '''
    #intialize using the outfile to infere other information
    out_file_folder, modelname, ResultFrame_initial, meta_dict, result_dict_file = initialize(outfile)
    #process the result frame, this can be done with or without groundTruth in order to calc things like majority votes
    ResultFrame_processed, nslices = processDataframe(out_file_folder, modelname, ResultFrame_initial, meta_dict)
    if testing:
        # when testing is set create metrics
        createMetrics(
            ResultFrame_processed, outfile, NumSlicesPerClass=nslices, modelname=modelname,
            meta_dict=meta_dict, certainties=certainties, result_dict_file=result_dict_file
        )
    if vis:
        #if vis is set creates visualized examples of different predictions for labels and datasets
        visualize_examples(out_file_folder, ResultFrame_processed, meta_dict)



if __name__=='__main__':
    #load the comand line arguments from argparse
    parser = argparse.ArgumentParser(description='This is the analysis of the results created in the model testing/prediction')
    parser.add_argument('-o','--outfile', action='store',metavar='o', required=True, help='pass here the filepath that was created by ModelTesting/predicting')
    parser.add_argument('-t','--testing',action='store_true', help='Pass this flag if testing should be done instead of only predictiong \n if testing results like accuracy and confusion matrices should be computed')
    parser.add_argument('-v', '--visualize', action='store_true', help='visualize examples of correclty and misspredicted classes')
    parser.add_argument('--certainties',action='store',metavar='cert', default=[], help='specify a comma septerated string of certainties that should be used for the testing has to be in [0,1]. This needs to have the -t flag to be set')
    args = parser.parse_args()
    out_file = args.outfile
    testing = args.testing
    certainties = args.certainties
    visualize = args.visualize
    #call the main function to analyse the results
    main(out_file, testing, certainties, visualize)
    print('finished analysing results')

