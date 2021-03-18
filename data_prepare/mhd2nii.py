import os
import SimpleITK as sitk

mhd_path = r'./scan'
ct_path = './sliver07_test/ct'
seg_path = './sliver07_test/label'
for i in os.listdir(mhd_path):
    if i.endswith('.mhd'):
        print(i)
        img = sitk.ReadImage(mhd_path+'/'+i)
        sitk.WriteImage(img,ct_path+'/'+i.split('.mhd')[0]+".nii.gz")