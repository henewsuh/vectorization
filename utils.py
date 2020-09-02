"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division #소수로 저장하는 모듈

import math
import pprint

import imageio
import numpy as np
import scipy.io
import scipy.misc
import cv2


file_path = "/home/hanew/module1/wall_objects_gt/"

# try:
#    _imread = scipy.misc.imread
# except AttributeError:
#    from imageio import imread as _imread

pp = pprint.PrettyPrinter() #예쁘게 출력해주는 거 

get_stddev = lambda x, k_h, k_w: 1 / math.sqrt(k_w * k_h * x.get_shape()[-1])


# -----------------------------

def image_show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey() 
    cv2.destroyWindow(name)
    
    
def load_test_data(image_path, fine_size=512):
    img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)   # Load image doing binarization 
    #img = scipy.misc.imresize(img, [fine_size, fine_size])  # Resize image to 1024 x 1024
    
    print(image_path)
    img = img.astype(np.float)  # change to numpy matrix format
    img /= 255.0
    img = cv2.resize(img, (fine_size, fine_size), interpolation = cv2.INTER_AREA)
    img = img.reshape(fine_size, fine_size, 1)  # Reshape 1024 x 1024 x 1
    return img
# 입력 이미지를 그레이스케일로 변환하고, 
# 1024 x 1024 x 1 사이즈로 조정 
    


def load_train_data(image_path, load_size=632, fine_size=512, is_testing=False):
    img_A = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    img_B = cv2.imread(image_path.replace('A', 'B'),cv2.IMREAD_GRAYSCALE)
    
    print(image_path)
    
    #img_B = imageio.imread(image_path.replace('A', 'B'), as_gray='T') #.replace(찾을값, 바꿀값, 바꿀횟수)

    if np.random.random() > 0.8: #values between 0 and 1 
        is_testing = True

    if not is_testing:
        
        #img_A = scipy.misc.imresize(img_A, [load_size, load_size])
        #img_B = scipy.misc.imresize(img_B, [load_size, load_size])
        
        img_A = cv2.resize(img_A, (load_size, load_size), interpolation = cv2.INTER_AREA)
        img_B = cv2.resize(img_B, (load_size, load_size), interpolation = cv2.INTER_AREA)
        
        h1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
        
        #low=0.01 high=120
        img_A = img_A[h1:h1 + fine_size, w1:w1 + fine_size]
        img_B = img_B[h1:h1 + fine_size, w1:w1 + fine_size]

    else:
    
        
        img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
        img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])
        #img_A = cv2.resize(img_A, (fine_size, fine_size), interpolation = cv2.INTER_AREA)
        #img_B = cv2.resize(img_B, (fine_size, fine_size), interpolation = cv2.INTER_AREA)
    
    img_A = img_A.astype(np.float)  # change to numpy matrix format
    img_A /= 255.0
    img_B = img_B.astype(np.float)  # change to numpy matrix format
    img_B /= 255.0 
    
    #img_A = (img_A < 255).astype(np.float)
    #img_B = (img_B < 255).astype(np.float)

    img_A = img_A.reshape(fine_size, fine_size, 1)
    img_B = img_B.reshape(fine_size, fine_size, 1)
    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB


# -----------------------------

def save_images(images, image_path):
    image = images[0]
    image = image.reshape(512, 512)
    image = (255-image)
    return scipy.misc.imsave(image_path, image)
