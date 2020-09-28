# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
from PIL import Image
import scipy.io as sio
from VGG16 import *
import math
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=50, help="number of images in batch")
parser.add_argument("--num_bit", type=int, default=12, help="number of hash bits")

parser.add_argument("--img_size", default=224, help="image size of input")
parser.add_argument("--checkpoint", default='./models-vgg/vgg16-12b-nus-unseen-top3-300d ', help="directory with checkpoint to resume training from or use for testing")
parser.add_argument("--model_name", default='vgg16-12b-nus-unseen-top3-300d', help="which model to be evaluated")
parser.add_argument("--img_dir", default="./dataset/NUS-WIDE/all_images/", help="directory of input images")
parser.add_argument("--img_file", default="./data/nus17/val_imagelists.txt", help="test image file")
parser.add_argument("--data_type", default="val", choices=["train", "test", "val"])
parser.add_argument("--output_dir", default="./results", help="where to put output images")
args = parser.parse_args()

mean_value = np.array([123, 117, 104]).reshape((1, 3))

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True


def toBinaryString(binary_like_values):
    numOfImage, bit_length = binary_like_values.shape
    list_string_binary = []
    for i in range(numOfImage):
        str = ''
        for j in range(bit_length):
            str += '0' if binary_like_values[i][j] <= 0 else '1'
        list_string_binary.append(str)
    return list_string_binary


def evaluate():
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    f = open(args.img_file)
    lines = f.readlines()
    l = len(lines)

    ckpt = tf.train.get_checkpoint_state(args.checkpoint)
    saver = tf.compat.v1.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')

    graph = tf.get_default_graph()

    hash_layer = graph.get_tensor_by_name('fc8/BiasAdd:0')
    image = graph.get_tensor_by_name('Placeholder_1:0')
    is_training = graph.get_tensor_by_name('Placeholder:0')

    with tf.Session(config=config) as sess:
        saver.restore(sess, ckpt.model_checkpoint_path)
        res = open(args.output_dir+'result_'+args.model_name+'_'+args.data_type+'.txt', 'w')
        for i in range(int(math.ceil(float(l)/args.batch_size))):
            print(i)
            data = np.zeros([args.batch_size, args.img_size, args.img_size, 3], np.float32)
            for j in range(args.batch_size):
                if j + i * args.batch_size < l:
                    try:
                        img_name = lines[j + i * args.batch_size].strip('\n\r')
                        img_path = args.img_dir + img_name
                        img = Image.open(img_path)
                        img = img.resize((args.img_size, args.img_size))
                        new_im = img - mean_value
                        new_im = new_im.astype(np.int16)
                        data[j, :, :, :] = new_im
                    except:
                        img_name = lines[1].strip('\n\r')
                        img_path = args.img_dir + img_name
                        img = Image.open(img_path)
                        img = img.resize((args.img_size, args.img_size))
                        new_im = img - mean_value
                        new_im = new_im.astype(np.int16)
                        data[j, :, :, :] = new_im
            t1=time.clock()
            eval_sess = sess.run(hash_layer, feed_dict={image: data, is_training:False})
            w_res = toBinaryString(eval_sess)
            print(time.clock()-t1)
        res.close()


if __name__ == '__main__':
    evaluate()





