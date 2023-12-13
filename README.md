# DeepDicomSort for Brain MRI

## origin
DeepDicomSort was originally created by van der Voort et all to recognize different MRI scan types (pre-contrast T1-weighted, post-contrast T1-weighted, T2-weighted, Proton Density-weighted, T2w-FLAIR-weighted, diffusion weighted imaging, perfusion weighted DSC and derived imaging) of brain tumor scans.
It does this using a CNN.

For more information, see the paper: https://doi.org/10.1007/s12021-020-09475-7

If you use DeepDicomSort in your work, please cite the following: van der Voort, S.R., Smits, M., Klein, S. et al. DeepDicomSort: An Automatic Sorting Algorithm for Brain Magnetic Resonance Imaging Data. Neuroinform 19, 159–184 (2021). https://doi.org/10.1007/s12021-020-09475-7

## adaptation
During his Internship Project Soenke van Loh adapted and retrained the DDS network in order to be able to recognize more different MRI scantypes and make it more applicable to clinical data.
DDS for Brain MRI comes with two different models, one with 16 different classes and one with eight different classes.

Model 1 (no binning):
T1,T1_c,T2,T2-FLAIR,PD,SWI,GRE,T2*,DWI,ADC,BOLD,angio,PWI,ASL,DTI,Other

Model 2 (alternative binning):
diffusion,perfusion,suscept,T1,T1_c,T2,T2-FLAIR,PD,Other

The Other class includes different scantypes like fieldmaps, localizers or phase scans.
## Set-up
DeepDicomSort requires the following:
- Python 3.9.5

After installing python 3.9.5, install the required packages:
`pip install -r requirements.txt`

## Running DeepDicomSort for Brain MRI

There are multiple ways DeepDicomSort for Brain MRI can be used. The most important parameters are set using a config file.
An example config file is provided. I advise to copy it and adapt it for the usecases. Additional parameters can or need to be set 
in the command line when running a file. Use the -h option when in doubt about which are needed or what they mean.

### prediction from dicom data
The dataset needs to be in one root folder where all dicom data is in one folder. This folder can contain sub folders.
In the config.yaml file all parameters in the preprocessing and model section need to be set by the user.
Preprocessing of the data is automatically done in the pipeline.
Then the network can be run using `python3 Model_EndToEndPrediction_cleaned.py -c 'config.yaml'`

### preprocessing data
For preprocessing of data all files are in the Preprocessing_cleaned folder. Different steps of the preprocessing
can be run using the according arguments in combination with the `preprocessing_pipeline_monai.py` file
The full pipleine applies the follwing steps to the raw dicom data:

Sorting to structured folder, splitting in series, converting to nifti, creating individual slices per scan

A csv is created which documents the according file path after sorting, splitting and converting to nifti aswell as if the data was successfully split.

### (transfer) training a new model
If annotated data is present in a tab seperated txt file where each row is one nifti slice in the first column and 
the numeric label in the second column then a new model can be trained or one of the base modesl can be tuned.
For this also all parameters in the training section need to be set. For transfer training from a previous model that
model can be set in the transfer weights part. Be cautious that the model needs to have the same number of output classes
or the lasy layers need to be manually removed. Meaning it is aso possible to only give the convolution layers of a model.
The dimensions for every layer that is transfered from need to match the base model.

### overview of the different python files
All files besides Model_analysis take their inputs from the config file. Often additonal parameters can be set. See the individual files for more information.
- Model_analysis: takes the output file of model testing or model predicting as input and creates majority votes or if 
ground truth labels are available also metrics can be calculated.
- Model_testing: Uses a test file (same structure as a training file) and a model as inputs in order to calculate predictions
with ground truth. 
- Model_predicting: Uses a folder of nifti slices and a model as input and predicts their class
- Model_training_new: Uses a train file and optionally transfer weights in order to train a new model
- Model_EndToEndPrediction_cleaned: Runs the full pipeline on raw dicom data
- Preprocessing_cleaned/preprocessing_pipeline_moani: This file can be used to run the full preprocessing pipeline or parts of it on data.

