import os

from monai.transforms import LoadImage, EnsureChannelFirst, SaveImage
import argparse
import numpy as np
import matplotlib.pyplot as plt

def saveImageToPng(image_path, out_dir):
    image = LoadImage(image_only=True)(image_path)
    img_shape = np.array(image).shape
    center_slide = int(img_shape[-1]/2)
    data = image[:, :, center_slide]
    plt.imshow(data[:,:],cmap='gray')
    filename = os.path.split(image_path)[1]
    filename = os.path.splitext(os.path.splitext(filename)[0])[0]
    outfile = os.path.join(out_dir, f'{filename}.png')
    plt.savefig(outfile)
    plt.close('all')
    print(f'saved image  {image_path} to {outfile}')

parser = argparse.ArgumentParser(description='saves a dicom file, dicom folder or nifti file to a png image')
parser.add_argument('-f','--file', action='store', help='pass here the file to save')
parser.add_argument('-o', '--outdir', action='store', help='where should the image be stored?')
parser.add_argument('-m', '--multiple_files', action='store_true', help='if this is set the file needs to\
                                        be a file of a certain format containing folder names and paths to images')
args = parser.parse_args()
image_path_files = ""
if args.multiple_files:
    image_path_files = args.file
else:
    image_path = args.file
out_dir = args.outdir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if image_path_files:
    with open(image_path_files, 'r') as image_file:
        Lines = image_file.readlines()
    for line in Lines:
        if 'outfolder' in line:
            temp_out_dir = os.path.join(out_dir, line.split('"')[1])
            print(temp_out_dir)
        else:
            print(line)
            saveImageToPng(line,out_dir)

else:
    saveImageToPng(image_path, out_dir)

