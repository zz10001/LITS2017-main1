# -*- coding: utf-8 -*-

import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from skimage import measure
from scipy.ndimage import label

import glob
from time import time

import copy
import math
import argparse
import random
import warnings
import datetime

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision
import ttach as tta

from dataset.dataset import Dataset

from net import Unet,sepnet,MultiResUnet,dense_UNet,segnet,UNetplusplus,cbam_Unet
from utilities.utils import str2bool, count_params
import joblib
import imageio
#import ttach as tta

test_ct_path = '../LITS2017/LITS2017_test'   #需要预测的CT图像
seg_result_path = './LITS2017/LITS2017_seg' #需要预测的CT图像标签，如果要在线提交codelab，需要先得到预测过的70例肝脏标签

pred_path = 'pred_result/pred_test2017_Unet'

if not os.path.exists(pred_path):
    os.mkdir(pred_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default=None,
                        help='')
    parser.add_argument('--training', type=bool, default=False,
                    help='whthere dropout or not')

    args = parser.parse_args()

    return args


def main():
    val_args = parse_args()

    args = joblib.load('models/LiTS_UNet_lym/2020-12-27-10-57-59/args.pkl')
    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')
    joblib.dump(args, 'models/LiTS_UNet_lym/2020-12-27-10-57-59/args.pkl')

    # create model
    print("=> creating model %s" %args.arch)
    model = Unet.U_Net(args)

    model = torch.nn.DataParallel(model).cuda()

    model.load_state_dict(torch.load('models/LiTS_UNet_lym/2020-12-27-10-57-59/epoch233-0.9736-0.8648_model.pth'))
    model.eval()
    # model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
    
    for file_index, file in enumerate(os.listdir(test_ct_path)):
        start = time()

        # if file.replace('volume', 'segmentation').replace('nii','nii.gz') in os.listdir(pred_path):
        #     print('already predict {}'.format(file))
        #     continue
        # 将CT读入内存
        ct = sitk.ReadImage(os.path.join(test_ct_path, file), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)

        mask = sitk.ReadImage(os.path.join(seg_result_path, file.replace('volume', 'segmentation')), sitk.sitkUInt8)
        mask_array = sitk.GetArrayFromImage(mask)

        mask_array[mask_array > 0] = 1

        print('start predict file:',file)

        ct_array[ct_array > 200] = 200
        ct_array[ct_array < -200] = -200

        ct_array = ct_array.astype(np.float32)
        ct_array = ct_array / 200

        # 找到肝脏区域开始和结束的slice，并各向外扩张slice
        z = np.any(mask_array, axis=(1, 2))
        start_slice, end_slice = np.where(z)[0][[0, -1]]

        start_slice = max(0, start_slice - 10)
        end_slice = min(mask_array.shape[0]-1, end_slice + 10)

        ct_crop = ct_array[start_slice:end_slice+1,32:480,32:480]

        slice_predictions = np.zeros((ct_array.shape[0],512,512),dtype=np.int16)

        with torch.no_grad():
            for n_slice in range(ct_crop.shape[0]-3):
                ct_tensor = torch.FloatTensor(ct_crop[n_slice: n_slice + 3]).cuda()
                ct_tensor = ct_tensor.unsqueeze(dim=0)
                # print('ct_tensor',ct_tensor.shape,n_slice)
                output = model(ct_tensor)
                output = torch.sigmoid(output).data.cpu().numpy()
                probability_map = np.zeros([1, 448, 448], dtype=np.uint8)
                #预测值拼接回去
                # i = 0
                for idz in range(output.shape[1]):
                    for idx in range(output.shape[2]):
                        for idy in range(output.shape[3]):
                            if (output[0,0, idx, idy] > 0.65):
                                probability_map[0, idx, idy] = 1        
                            if (output[0,1, idx, idy] > 0.5):
                                probability_map[0, idx, idy] = 2

                slice_predictions[n_slice+start_slice+1,32:480,32:480] = probability_map        

            pred_seg = slice_predictions
            pred_seg = pred_seg.astype(np.uint8)

            pred_seg = sitk.GetImageFromArray(pred_seg)

            pred_seg.SetDirection(ct.GetDirection())
            pred_seg.SetOrigin(ct.GetOrigin())
            pred_seg.SetSpacing(ct.GetSpacing())

            sitk.WriteImage(pred_seg, os.path.join(pred_path, file.replace('volume', 'segmentation').replace('nii', 'nii.gz')))

            speed = time() - start

            print(file, 'this case use {:.3f} s'.format(speed))
            print('-----------------------')

            torch.cuda.empty_cache()
                        
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main()
            

        
