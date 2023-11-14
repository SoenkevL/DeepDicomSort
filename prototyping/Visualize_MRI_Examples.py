import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
import os


visualize_frame = pd.read_csv('visualizeFrame.csv')
def saveNifti(img_path, outfile):
    nimg = nib.load(img_path)
    data = nimg.get_fdata()
    plt.figure()
    plt.imshow(data[:,:],cmap='gray')
    plt.savefig(outfile)
    plt.close('all')
def createMRIexamples(vis_frame_row, output_folder):
    outpath = os.path.join(output_folder, vis_frame_row['string_label'], vis_frame_row['dataset'])
    counter = 0
    if not os.path.exists(outpath):
        counter = 1
        os.makedirs(outpath)
    outfile = os.path.join(outpath, f"{vis_frame_row['string_label']}__{vis_frame_row['dataset']}__{counter}.png")
    saveNifti(vis_frame_row['ID'], outfile)
visualize_frame.apply(createMRIexamples, output_folder='', axis=1)