import xnat

from Pytorch_monai.secrets import xnatpw

with xnat.connect('https://xnat.bmia.nl',user='svanloh',password=xnatpw) as session:
    project = session.projects['ace']
    # SubjectFrame = project.subjects.tabulate_pandas()
    # SubjectFrame.to_csv('Subjects.csv')
    # ExperimentFrame = project.subjects[0].experiments.tabulate_pandas()
    # ExperimentFrame.to_csv('Experiemts.csv')
    # ScansFrame = project.subjects[0].experiments[0].scans.tabulate_pandas()
    # ScansFrame.to_csv('Scans.csv')
    # download_path = ScansFrame[ScansFrame['type']=='T1']
    for subject in project.subjects:
        ID = project.subjects[subject].label
        print(ID)
        try:
            project.subjects[subject].experiments[0].scans['T1'].download(f'/trinity/home/r098375/DDS/XNAT/downloads/{ID}_T1.zip')
        except:
            continue
