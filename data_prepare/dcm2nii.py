# for task2 chaos 

import os
import SimpleITK as sitk
res=[]
for filename in os.listdir(r'CT'):
    # print(filename+'/DICOM_anon')
    res.append(filename+'/DICOM_anon')
# print(res)
for i in res:
    file_path = os.path.join('CT',i) #dicom存放文件夹
    print(i.split('/')[0])
    print(file_path)
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(file_path)
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(file_path)

    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)

    image3D = series_reader.Execute()
    sitk.WriteImage(image3D, i.split('/')[0]+'.nii.gz')
