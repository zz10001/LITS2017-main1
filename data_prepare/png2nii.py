import SimpleITK as sitk
import glob
import numpy as np
from PIL import Image
import cv2
import glob
 
import matplotlib.pyplot as plt # plt 用于显示图片
 
import os
res=[]
niipath = 'testct'
new_seg = 'testseg'
for filename in os.listdir(r'CT'):
    # print(filename+'/DICOM_anon')
    res.append(filename+'/Ground')
# print(res)
for i in res:
    file_path = os.path.join('CT',i) #png存放文件夹
    idx = i.split('/')[0]
    # print(idx,i)
    png_path = glob.glob('./CT/'+idx+'/Ground/*')
    png_path.sort(reverse=True)
    # print(png_path)
    niip = os.path.join(niipath,idx+'.nii.gz')
    # print(niip)
    ct = sitk.ReadImage(niip, sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)
    print(ct_array.shape)
    # for i in range(ct_array.shape[0]):
    #     plt.imshow(ct_array[i,:,:])
    #     # cv2.imwrite(target+'/'+'30'+str(i)+".png",ct_array[i,:,:])
    #     plt.show()
    seg_array = np.zeros(ct_array.shape, dtype='uint8')
    for i,d in enumerate(png_path):
        # print(i,d)
        img_as_img = Image.open(d)
        # img_as_np = np.asarray(img_as_img)
        # allImg[i, :, :] = img_as_np
        # img_as_img.show()
        seg_array[i,:,:] = img_as_img

    new_seg = sitk.GetImageFromArray(seg_array)
    new_seg.SetDirection(ct.GetDirection())
    new_seg.SetOrigin(ct.GetOrigin())
    new_seg.SetSpacing(ct.GetSpacing())
    sitk.WriteImage(new_seg, idx+'.nii.gz')

