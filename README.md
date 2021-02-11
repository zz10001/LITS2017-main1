**2d segmentation for LiTS2017** (the challenge website in [codalab](https://competitions.codalab.org/competitions/17094))
--
**data**:[GoogleDrive](https://drive.google.com/drive/folders/1V6X3CwnHMoVyuArASiNgoOcC5N4oNpLA?usp=sharing)
# How to use it 
step 1 data process
-   
``python data_prepare/preprocess.py``

you can get the data like this,each npy file shape is ``448*448*3``,use each slice below and above as input image,the mid slice as mask(``448*448*1``)
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
``python data_prepare/python preprocess.py``
# baseline

| Method     |U-Net  |R2U-Net|Att U-Net|Att R2U-Net |
| :----------:|:----:| :-----:|:-------:|:--------:|
| `Dice(liver)`|0.951|0.953  |0.953    |0.953    |
| `rvd`        |0.016|0.016  |0.016    |0.016    |
| `jaccard`    |0.911|0.911  |0.911    |0.911    |
| `Dice(tumor)`|0.613|0.623  |0.623    |0.623    |
| `rvd`        |-0.076| -0.076|-0.076   |-0.076   |
| `jaccard`    |0.634|0.634  |0.634    |0.634    |

the code is built on [Image_Segmentation](https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py)

# Later work
- [ ] data augmentation
- [ ] only segment tumor 
- [ ] postprocessing
- [ ] 3d segmentation