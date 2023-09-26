from pathlib import Path
import xnat
from xnat.exceptions import XNATResponseError
import os
import csv
import time




def download_assessor(xnat_project, subject_label, experiment_label,
                      assessor_label, download_dir):
    try: 
        xnat_subject = xnat_project.subjects[subject_label]
        xnat_experiment = xnat_subject.experiments[experiment_label]
        if not xnat_experiment:
            print(f"Skipping {xnat_subject.label} failed to find experiment")
            return
        if assessor_label in xnat_experiment.assessors:
            xnat_assessor = xnat_experiment.assessors[assessor_label]
            #print('xnat_assessor:',xnat_assessor)
            #print('xnat_assessor.label:',xnat_assessor.label)
        else:
            print(f"Skipping {xnat_subject.label} failed to find assessor")
            return
        for resource_idx, resource_label in enumerate(xnat_assessor.resources):
		    #screen on the resource in a experiment
            xnat_resource = xnat_assessor.resources[resource_label]
            #print("resource_idx: {}".format(resource_idx))
            # print(xnat_resource.files)
            for idx, filename in enumerate(xnat_resource.files):
                xnat_file = xnat_resource.files[idx]
                print('xnat_file',xnat_file)
                if xnat_file.path == 'Template_space/Brain_image_in_MNI_space/result.nii.gz':
				    #this is the specific file I need to download, it can be different in each job
                    print(xnat_file.path)
                    experiment_label_edit = experiment_label.replace(experiment_label[10],'.')
					#this is related to the naming of each subject
                    download_dir = download_dir / f"{subject_label}" / f"{experiment_label_edit}"
					#structure you download direction file, I create a separate file for each subject
                    if os.path.exists(download_dir):
                        print("{}already exist".format(subject_label))
                    else:
                        download_dir.mkdir(parents=True, exist_ok=True)
                        download_path = download_dir / "T1w.nii.gz"
                        #print('download_path:',download_path)
                        xnat_file.download(str(download_path))
                        
    except KeyError as e:
        print(f"ERROR downloading {experiment_label} reason: {e}")
    except XNATResponseError as e:
        print(f"ERROR downloading {experiment_label} reason: {e}")
        
def read_subjects_from_csv(csv_file):
    f_content = open(csv_file, "r",encoding='UTF-8-sig')			
    r = csv.DictReader(f_content)		
    PSI_ids=[]
    for row in r:
        PSI_id=row['PTID']
        PSI_id = PSI_id.replace('.','_')
        PSI_ids.append(PSI_id)
        print(PSI_id)
    return PSI_ids



            
def main():
    start = time.perf_counter()
    print(start)
    download_dir = Path('')
    label_dir=Path('')
    host = 'https://bigr-rad-xnat.erasmusmc.nl/'
    user=''
    password=''
    project = 'ADNIatBIGR'
    assessor_tag = 'iris_template'
    csv_file=os.path.join("/trinity/home/wkang/Xnat/ADNI/label/subjectlist_train.csv")
	#I need to download data for specified subjects, and I have a csv file for the subject number
    PSI_ids=read_subjects_from_csv(csv_file)
	#function to screen subject ID in csv file (I didn't use it in this script)

    print("host: {}".format(host))
    print("project: {}".format(project))
    print("assessor_tag: {}".format(assessor_tag))
    
    with xnat.connect(host,user,password) as xnat_host:
	#connect to the server
        xnat_project = xnat_host.projects[project]
        for subject_id in xnat_project.subjects:
		#loop to find the data
            #print('subject_id',subject_id)
            xnat_subject = xnat_project.subjects[subject_id]
            #print('xnat_subject',xnat_subject)
            for experiment_id in xnat_subject.experiments:
                xnat_experiment = xnat_subject.experiments[experiment_id]
                #print(xnat_experiment.label)
                #print("xnat_experiment: {}".format(xnat_experiment))
                assessor_label = f"{xnat_experiment.label}_{assessor_tag}"
                #print('xnat_subject.label:',xnat_subject.label)
                if f"{xnat_experiment.label}" in PSI_ids:
                    if os.path.exists(download_dir / f"{xnat_subject.label}" / f"{xnat_experiment.label}"):
                        print("{}already exist".format(xnat_subject.label))
                        continue
                    else:
                        download_assessor(xnat_project, xnat_subject.label, xnat_experiment.label, assessor_label,download_dir)
						#function to download data
                else:
                 print("not in list:",xnat_subject.label)
                 continue
            #break， you can try to download data of one subject at first 
        end = time.perf_counter()
        time_consume=end-start
        with open('time.txt','w') as f:
          f.write(str(time_consume))
        
          
                                  

if __name__ == "__main__":
    main()
