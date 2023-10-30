#based on python 3.9.5
import pandas as pd
from tqdm import tqdm
import xnat
import os
import argparse

def createDataFrame(server, user, password, project, outputFolder='', save_Individual_csv=False):
    if outputFolder == '':
        outputFolder = os.getcwd()
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    if not 'https://' in server:
        server = 'https://'+ server
    print(f'creating csv for project {project} on server {server}')
    with xnat.connect(server,user=user,password=password) as session:
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
        FullFrame.to_csv(os.path.join(outputFolder, 'RSS/FullFrame_RSS.csv'))
        return FullFrame

def cleanupFrame(dataframe, dropna=True, dropna_thresh=0.9, onlyScans=True):
    CleanedFrame = dataframe.drop(['subjectId','experimentId'],axis=1)
    if dropna:
        CleanedFrame = CleanedFrame.dropna(axis=1, thresh=dropna_thresh)
    if onlyScans:
        CleanedFrame = CleanedFrame[CleanedFrame['xnat_imagescandata_id'].notna()]
    CleanedFrame = CleanedFrame.rename({'ID':'ID_scan', 'URI':'URI_scan'}, axis=1)
    return CleanedFrame

def download_experiments(dataframe,dirname, server, user, pw):
    downloadKeys = dataframe['ID_exp'].unique()
    with xnat.connect(server,user=user,password=pw) as session:
        for expid in downloadKeys:
            experiment = session.experiments[expid]
            experiment.download_dir(dirname)

def main():
    parser = argparse.ArgumentParser('provide arguments which xnat server and project you are interested in')
    parser.add_argument('--path', '-p', action='store', default='', help='provide the output path the program should use. If none is given it will try to save to the current working directory')
    parser.add_argument('--server', '-s', action='store', required=True, help='which xnat server should be used to find the project')
    parser.add_argument('--Project', '-P', action='store', required=True, help='which project should be listed in the csv')
    parser.add_argument('--user', '-u', action='store', required=True, help='give a valid username for the server u are trying to access')
    parser.add_argument('--individual_csvs', '-i', action='store_true', help='add this flag if you want to save the inbetween steps like a frame with only all subjects, experiments and scans')
    parser.add_argument('--password','-pw', action='store', required=True, help='give the password for your username')
    arg = parser.parse_args()
    FullFrame = createDataFrame(arg.server, arg.user, arg.password, arg.Project, arg.path, arg.individual_csvs)
    cleanupFrame = cleanupFrame(FullFrame)
    cleanupFrame.to_csv(os.path.join(arg.path,'CleanedFrame.csv'))

if __name__=='__main__':
    main()

