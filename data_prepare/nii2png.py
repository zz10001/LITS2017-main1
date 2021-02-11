import SimpleITK as sitk
import scipy.io
import os
import natsort
import time
import cv2 as cv
import scipy.misc

start = time.time()
nii_path = '../LITS2017/CT'
save_path = '../png/ct'

folders = natsort.natsorted(os.listdir(nii_path))
for i in range(len(folders)):
  folders_path = os.path.join(nii_path, folders[i])
  img = sitk.ReadImage(folders_path)
  img_array = sitk.GetArrayFromImage(img)
  z = img_array.shape[0]
  for j in range(z):
    silce = img_array[j,:,:]
    cv.imwrite(os.path.join(save_path, str(i))+'/'+str(j+1)+'.png', silce)
    end = time.time()
    print(end-start)