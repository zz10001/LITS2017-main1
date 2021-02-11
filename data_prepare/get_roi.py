
import cv2 
import os
import time
import shutil
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")


ct_dir = '/home/amax/SDB/lym/LITS2017/CT'
seg_dir = '/home/amax/SDB/lym/LITS2017/seg'

new_ct_dir2 = 'train_cs/ct'
new_seg_dir2 = 'train_cs/seg'

upper = 200
lower = -200


def find_bb(volume):
	img_shape = volume.shape
	bb = np.zeros((6,), dtype=np.uint)
	bb_extend = 3
	# axis
	for i in range(img_shape[0]):
		img_slice_begin = volume[i,:,:]
		if np.sum(img_slice_begin)>0:
			bb[0] = np.max([i-bb_extend, 0])
			break

	for i in range(img_shape[0]):
		img_slice_end = volume[img_shape[0]-1-i,:,:]
		if np.sum(img_slice_end)>0:
			bb[1] = np.min([img_shape[0]-1-i + bb_extend, img_shape[0]-1])
			break
	# seg
	for i in range(img_shape[1]):
		img_slice_begin = volume[:,i,:]
		if np.sum(img_slice_begin)>0:
			bb[2] = np.max([i-bb_extend, 0])
			break

	for i in range(img_shape[1]):
		img_slice_end = volume[:,img_shape[1]-1-i,:]
		if np.sum(img_slice_end)>0:
			bb[3] = np.min([img_shape[1]-1-i + bb_extend, img_shape[1]-1])
			break

	# coronal
	for i in range(img_shape[2]):
		img_slice_begin = volume[:,:,i]
		if np.sum(img_slice_begin)>0:
			bb[4] = np.max([i-bb_extend, 0])
			break

	for i in range(img_shape[2]):
		img_slice_end = volume[:,:,img_shape[2]-1-i]
		if np.sum(img_slice_end)>0:
			bb[5] = np.min([img_shape[2]-1-i+bb_extend, img_shape[2]-1])
			break
	
	return bb

def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

def window_transform(ct_array, windowWidth, windowCenter, normal=False):
    """
    return: trucated image according to window center and window width
    and normalized to [0,1]
    """
    minWindow = float(windowCenter) - 0.5*float(windowWidth)
    newimg = (ct_array - minWindow) / float(windowWidth)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    if not normal:
        newimg = (newimg * 255).astype('uint8')
    return newimg

def Z_ScoreNormalization(x,mu,sigma):  
	x = (x - mu) / sigma 
	return x


for ct_file in os.listdir(ct_dir):

    # 将CT和金标准入读内存
    ct = sitk.ReadImage(os.path.join(ct_dir, ct_file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)
    spacing = ct.GetSpacing()
    spacing = np.array(spacing)
    print('------ct_file------',ct_file)
    print('spacing',spacing)
    seg = sitk.ReadImage(os.path.join(seg_dir, ct_file.replace('volume', 'segmentation')), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)

    # ct_array = window_transform(ct_array,200,50, normal=False)
    # ct_array = clahe_equalized(ct_array)
    # ct_array = ct_array/255
    # 数据处理
    ct_array[ct_array > upper] = upper
    ct_array[ct_array < lower] = lower

    ct_array = Z_ScoreNormalization(ct_array,np.average(ct_array),np.std(ct_array))		
		
    print('window transform:',ct_array.min(),ct_array.max())
    

    pred_liver = seg_array.copy()
    pred_liver[pred_liver>0] = 1

    bbox = get_bbox_from_mask(pred_liver,outside_value=0)
    bb = find_bb(pred_liver)

    print('bbox',bbox)
    print('bb',bb,bb[0],bb[3])
    print('ct',ct_array.shape)
    print('\n')

    # new_ct_array=ct_array[bbox[0][0]:bbox[0][1],bbox[1][0]:bbox[1][1],bbox[2][0]:bbox[2][1]]
    # new_seg_array = seg_array[bbox[0][0]:bbox[0][1],bbox[1][0]:bbox[1][1],bbox[2][0]:bbox[2][1]]

    # # new_ct_array = ct_array[46:83,200:456,190:510]
    # # new_seg_array = seg_array[46:83,200:456,190:510]
    # # seg_array[seg_array==1] = 0
    # # seg_array[seg_array==2] = 1

    # new_ct = sitk.GetImageFromArray(new_ct_array)

    # new_ct.SetDirection(ct.GetDirection())
    # new_ct.SetOrigin(ct.GetOrigin())
    # new_ct.SetSpacing(ct.GetSpacing())

    # new_seg = sitk.GetImageFromArray(new_seg_array)

    # new_seg.SetDirection(ct.GetDirection())
    # new_seg.SetOrigin(ct.GetOrigin())
    # new_seg.SetSpacing(ct.GetSpacing())

    # new_ct_name = 'volume-' + str(77) + '.nii.gz'
    # new_seg_name = 'segmentation-' + str(77) + '.nii.gz'

    # sitk.WriteImage(new_ct, os.path.join(new_ct_dir2, new_ct_name))
    # sitk.WriteImage(new_seg, os.path.join(new_seg_dir2, new_seg_name))










