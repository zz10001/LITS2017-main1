**2d segmentation for LiTS2017** (the challenge website in [codalab](https://competitions.codalab.org/competitions/17094))
--
**data**:  
[GoogleDrive](https://drive.google.com/drive/folders/1V6X3CwnHMoVyuArASiNgoOcC5N4oNpLA?usp=sharing)  
[百度云](https://pan.baidu.com/s/1leTOp_HWoZZ3YKnRlRQa-w)(7itj)
# How to use it 
step 1 data process
-   
``python data_prepare/preprocess.py``

you can get the data like this,each npy file shape is ``448*448*3``,use each slice below and above as input image,the mid slice as mask(``448*448``)
```
data---
    trainImage_k1_1217---
        1_0.npy
        1_1.npy
        ......
    trainMask_k1_1217---
        1_0.npy
        1_1.npy
        ......
```
step 2 Train the model 
--
``python train.py``

step 3 Test the model 
--
``python test.py``

step 4 postprocess to remove False predict
--
``python data_prepare/postprocess.py``
# baseline
| Method     |U-Net  |Att U-Net|sep U-Net |denseunet  |
| :----------:|:----:| :-----:|:-------:|:--------:|
| `Dice(liver)`|0.951|0.950  |0.948    | 0.949    |
| `rvd`        |0.016|0.038  |0.037    |0.029 |
| `jaccard`    |0.911|0.906  |0.903    |0.904    |
| `Dice(tumor)`|0.613|0.609  |0.594    |0.600    |
| `rvd`        |-0.076| -0.067|-0.096   |-0.119   |
| `jaccard`    |0.634|0.621  |0.604    |0.614    |

the code is built on [Image_Segmentation](https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py)

# Later work
- [ ] data augmentation
- [ ] postprocessing
- [ ] 3d segmentation

# Some others Related Work on Liver tumor segmentation
**3d model for segmentation**
[3DUNet-Pytorch](https://github.com/lee-zq/3DUNet-Pytorch)  
[MICCAI-LITS2017](https://github.com/assassint2017/MICCAI-LITS2017)  
[RAUNet-tumor-segmentation](https://github.com/RanSuLab/RAUNet-tumor-segmentation)  
[LiTS---Liver-Tumor-Segmentation-Challenge](https://github.com/junqiangchen/LiTS---Liver-Tumor-Segmentation-Challenge)  
[H-DenseUNet](https://github.com/xmengli999/H-DenseUNet)  

**2d model for segmentation**
[LiTS-Liver-Tumor-Segmentaton-Challenge](https://github.com/ChoiDM/LiTS-Liver-Tumor-Segmentaton-Challenge)  
[unet-lits-2d-pipeline](https://github.com/Confusezius/unet-lits-2d-pipeline)  
[DS-SFFNet](https://github.com/LTYUnique/DS-SFFNet)  


