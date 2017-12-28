import numpy as np
import os
from PIL import Image, ImageOps
import argparse
import random
from scipy import misc


def PcaJittering(DirPath):
    if not os.path.exists(DirPath):
        print("please input corect path")
        return
    for subDirName in os.listdir(DirPath):
        subDirPath = os.path.join(DirPath, subDirName)
        if os.path.isdir(subDirPath):
            if subDirPath.find('_')==-1:
                continue
            #print(subDirPath)
            for ImageName in os.listdir(subDirPath):
                image_path = os.path.join(subDirPath, ImageName)
                if ImageName.count("_") > 1:
                    continue
                img = Image.open(image_path)
                img = np.asarray(img, dtype='float32')
                img /= 255
                img_size = img.size // 3
                img1 = img.reshape(img_size, 3)
                img1 = np.transpose(img1)
                img_cov = np.cov([img1[0], img1[1], img1[2]])
                lamda, p = np.linalg.eig(img_cov)
                p = np.transpose(p)
                alpha1 = random.normalvariate(0, 0.3)
                alpha2 = random.normalvariate(0, 0.3)
                alpha3 = random.normalvariate(0, 0.3)
                v = np.transpose(
                    (alpha1 * lamda[0], alpha2 * lamda[1], alpha3 * lamda[2]))
                add_num = np.dot(p, v)
                img2 = np.array(
                    [img[:, :, 0] + add_num[0], img[:, :, 1] + add_num[1], img[:, :, 2] + add_num[2]])
                img2 = np.swapaxes(img2, 0, 2)
                img2 = np.swapaxes(img2, 0, 1)
                newImageName = os.path.join(
                    subDirPath, ImageName.split('.')[0] + '_3.jpg')
                print(newImageName)
                misc.imsave(newImageName, img2)


if __name__ == "__main__":
    PcaJittering("train/")
