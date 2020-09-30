import numpy as np
from PIL import Image
import cv2


def data_load(filelst, labelfile, img_size, img_dir, mean_value):
    labels = np.load(labelfile)

    f1 = open(filelst)
    lines1 = f1.readlines()
    l = len(lines1)
    # l=int(l)

    if l >10000:
        l = 10000
        labels = labels[0:l, :]

    datas = np.zeros([l, img_size, img_size, 3], np.float32)
    for i in np.arange(l):
        img_name = lines1[i].strip('\n\r')
        img_path = img_dir + img_name
        img = Image.open(img_path)
        img = img.resize((img_size, img_size))
        img = img.convert('RGB')
        new_im = img - mean_value
        new_im = new_im.astype(np.int16)
        datas[i, :, :, :] = new_im
        if i % 1000 == 0:
            print("[%d/%d] images processed!" %(i, l))
    return datas, labels

def data_load_length(filelst, labelfile, img_size, img_dir, mean_value,length):
    labels = np.load(labelfile)

    f1 = open(filelst)
    lines1 = f1.readlines()
    # l = len(lines1)
    l=int(length)

    # if l >10000:
    #     l = 10000
    labels = labels[0:l, :]

    datas = np.zeros([l, img_size, img_size, 3], np.float32)
    for i in np.arange(l):
        img_name = lines1[i].strip('\n\r')
        img_path = img_dir + img_name
        img = Image.open(img_path)
        img = img.resize((img_size, img_size))
        img = img.convert('RGB')
        new_im = img - mean_value
        new_im = new_im.astype(np.int16)
        datas[i, :, :, :] = new_im
        if i % 1000 == 0:
            print("[%d/%d] images processed!" %(i, l))
    return datas, labels