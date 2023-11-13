#based on python 3.9.5
import pandas as pd
from tqdm import tqdm
import xnat
import os
import argparse

# def createDataFrame(server, user, password, project, outputFolder='', save_Individual_csv=False):
def createDataFrame(server, project, outputFolder='', save_Individual_csv=False):
    '''
    Creates a dataframe listing everything in a specific xnat project on a server.
    Requires the login data for the project and server to be saved in a netrc file
    Inputs:
    -server: the server the project is on
    -project: the project to list
    -outputFolder: where the dataframe should be saved
    -save_Individual_csv: If only one big dataframe or also subjects, experiments and scans as seperate dataframes
        should be saved, defaults to False.
    Outputs:
    -FullFrame: dataframe containing all the info from the project
    '''
    if outputFolder == '':
        outputFolder = os.getcwd()
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    if not 'https://' in server:
        server = 'https://'+ server
    print(f'creating csv for project {project} on server {server}')
    # with xnat.connect(server,user=user,password=password) as session:
    with xnat.connect(server) as session:
        project = session.projects[project]
        subjects = project.subjects
        SubjectFrame = subjects.tabulate_pandas()
        FullFrame = SubjectFrame
        ExperimentFrameAll = pd.DataFrame()
        ScansFrameAll = pd.DataFrame()
        for subjectId in tqdm(list(SubjectFrame['ID'])):
            subject = subjects[subjectId]
            try:
                experiments = subject.experiments
                ExperimentFrame = experiments.tabulate_pandas()
                ExperimentFrame['subjectId'] = subjectId
                if ExperimentFrameAll.empty:
                    ExperimentFrameAll = ExperimentFrame
                else:
                    ExperimentFrameAll = pd.concat([ExperimentFrameAll, ExperimentFrame])
            except:
                continue
            for experimentId in list(ExperimentFrame['ID']):
                experiment = experiments[experimentId]
                try:
                    scans = experiment.scans
                    ScansFrame = scans.tabulate_pandas()
                    ScansFrame['experimentId'] = experimentId
                    if ScansFrameAll.empty:
                        ScansFrameAll = ScansFrame
                    else:
                        ScansFrameAll = pd.concat([ScansFrameAll, ScansFrame])
                except:
                    continue
        if save_Individual_csv:
            SubjectFrame.to_csv(os.path.join(outputFolder, 'SubjectFrame.csv'))
            ExperimentFrameAll.to_csv(os.path.join(outputFolder, 'ExperimentFrame.csv'))
            ScansFrameAll.to_csv(os.path.join(outputFolder,'ScansFrame.csv'))
        FullFrame = FullFrame.merge(ExperimentFrameAll, how='left', left_on='ID', right_on='subjectId', suffixes=('_sub','_exp'))
        FullFrame = FullFrame.merge(ScansFrameAll, how='left', left_on='ID_exp', right_on='experimentId', suffixes=('','_scan'))
        FullFrame.to_csv(os.path.join(outputFolder, 'FullFrame.csv'))
        return FullFrame

def cleanup_frame(dataframe, dropna=True, dropna_thresh=0.9, onlyScans=True):
    '''
    creates a csv from the full dataframe created by createDataFrame which drops columns with too many Na values
    and only keeps annotated experiments with scans on default
    Inputs:
    -dataframe: needs to be the output of createDataframe or a frame of similar fashion.
    -dropna: should rows containing na be dropped
    -dropna_threshhold :can be used to specify how much na needs to be in a column to be dropped, defaults 0.9
    -onlyscans: makes sure that only experiments that contain scans will be kept
    -dropTypeUnknown: drops all experiments where the scans are not annotated and only have unknown in their description
    Outputs:
    - CleanedFrame: pandas dataframe after cleanup
    '''
    CleanedFrame = dataframe.drop(['subjectId','experimentId'],axis=1)
    if dropna:
        CleanedFrame = CleanedFrame.dropna(axis=1, thresh=dropna_thresh)
    if onlyScans:
        CleanedFrame = CleanedFrame[CleanedFrame['xnat_imagescandata_id'].notna()]
    CleanedFrame = CleanedFrame.rename({'ID':'ID_scan', 'URI':'URI_scan'}, axis=1)
    CleanedFrame = CleanedFrame[CleanedFrame['type'].str.lower().str.contains('unknown')!=True]
    return CleanedFrame


# def download_experiments(dataframe,dirname, server, user, pw):
def download_experiments(dataframe, dirname, server):
    '''
    will download all experiments in the csv into the dedicated directory
    inputs:
    -dataframe: the frame that should be used to download the experiments
    -dirname: where should the experiments be saved to (datapath)
    -server: the server where the experiments can be found (needs to have login data in netrc file)
    outputs:
    '''
    downloadKeys = dataframe['ID_exp'].unique()
    # with xnat.connect(server,user=user,password=pw) as session:
    with xnat.connect(server) as session:
        for expid in downloadKeys:
            try:
                experiment = session.experiments[expid]
                experiment.download_dir(dirname)
            except:
                print(f'could not download experiment: {expid}')

def main():
    """
    main function which uses argparse in order to fill in the arguments needed by createDataframe to create the dataframe for a given project on a server.
    """
    parser = argparse.ArgumentParser('provide arguments which xnat server and project you are interested in')
    parser.add_argument('--path', '-p', action='store', default='', help='provide the output path the program should use. If none is given it will try to save to the current working directory')
    parser.add_argument('--server', '-s', action='store', required=True, help='which xnat server should be used to find the project')
    parser.add_argument('--Project', '-P', action='store', required=True, help='which project should be listed in the csv')
    # parser.add_argument('--user', '-u', action='store', required=True, help='give a valid username for the server u are trying to access')
    parser.add_argument('--individual_csvs', '-i', action='store_true', help='add this flag if you want to save the inbetween steps like a frame with only all subjects, experiments and scans')
    # parser.add_argument('--password','-pw', action='store', required=True, help='give the password for your username')
    arg = parser.parse_args()
    FullFrame = createDataFrame(arg.server, arg.Project, arg.path, arg.individual_csvs)
    cleanupFrame = cleanup_frame(FullFrame)
    cleanupFrame.to_csv(os.path.join(arg.path,'CleanedFrame.csv'))

if __name__=='__main__':
    main()

