import numpy as np
import cv2
import matplotlib.pyplot as plt
import os 

image = np.load("trainImage_k2_1217/4_56.npy")
target = 'cs'

if not os.path.exists(target):
    os.makedirs(target)
print(image.shape)
print('max0',np.max(image),'min0',np.min(image))
for i in range(image.shape[2]):
    plt.imshow(image[:,:,i])
    cv2.imwrite(target+'/'+'4_56_'+str(i)+".png",image[:,:,i]*255)
    plt.show()

# plt.imshow(image)
# cv2.imwrite(target+'/'+ 'mm_4_56'+".png",image*127.5)
# plt.show()


