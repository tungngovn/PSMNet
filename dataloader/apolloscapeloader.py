import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):

    ## Folder for specific images
    left_fold  = 'camera_5/'
    right_fold = 'camera_6/'
    disp_noc   = 'disparity/'

    ## Add left images file names to image
    image = [img for img in os.listdir(filepath+disp_noc)] #if img.find('_10') > -1]

    ## Split to train and val sets
    train = image[:]
    val   = image[:214]

    ## Create list of train images
    left_train  = [filepath+left_fold+img[0: len(img) - 5]+'5.jpg' for img in train]
    right_train = [filepath+right_fold+img[0: len(img) - 5]+'6.jpg' for img in train]
    disp_train = [filepath+disp_noc+img for img in train]

    ## Create list of val images
    left_val  = [filepath+left_fold+img[0: len(img) - 5]+'5.jpg' for img in val]
    right_val = [filepath+right_fold+img[0: len(img) - 5]+'6.jpg' for img in val]
    disp_val = [filepath+disp_noc+img for img in val]

    return left_train, right_train, disp_train, left_val, right_val, disp_val
