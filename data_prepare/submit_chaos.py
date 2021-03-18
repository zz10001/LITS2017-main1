# make submit on https://chaos.grand-challenge.org/evaluation/challenge/submissions/create/


import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
import os 
import SimpleITK as sitk
from skimage import morphology, measure, io

# image = np.load("10_113.npy")
target = 'cs/3/Results'

if not os.path.exists(target):
    os.makedirs(target)

def getLargestCC(segmentation):
    labels = measure.label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

ct = sitk.ReadImage('3.nii.gz', sitk.sitkUInt8)
ct_array = sitk.GetArrayFromImage(ct)
ct_array = np.uint8(getLargestCC(ct_array>0))*255
print(ct_array.shape)
slice_num = ct_array.shape[0]
for i in range(ct_array.shape[0]):
    # plt.imshow(ct_array[i,:,:])
    save_name = 'img' + ("%03d" % i) + '.png'
    cv2.imwrite(os.path.join(target,save_name),ct_array[slice_num-i-1,:,:])
    plt.show()
