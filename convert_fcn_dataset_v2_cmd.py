#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import cv2
import numpy as np
import tensorflow as tf
from vgg import vgg_16
from object_detection.utils import dataset_util
import sys


'''
功能:将数据集转成tensorflo的tfrecord格式

分片生成多个文件.


fcn_train_00000of00002.record
fcn_train_00001of00002.record
fcn_train_00002of00002.record

fcn_val_00000of00002.record
fcn_val_00001of00002.record
fcn_val_00002of00002.record


注意: 该文件必须在命令行下执行才能生效, 如果在IDE, 比如 pycharm上执行会出错

cd到该文件所在目录
python convert_fcn_dataset_v2_cmd.py
'''


'''
通过脚本了来执行该文件
注意传入的数据地址以及输出地址
python convert_fcn_dataset_v1_sh.py --data_dir=./VOC2012 --output_dir=./

flags部分是脚本名来执行使用
'''
# flags = tf.app.flags
# flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
# flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')

data_dir = '../datasets/raw/VOC2012'
output_dir = '../datasets/train'



classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']
# RGB color for each class
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [
                128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]

cm2lbl = np.zeros(256 ** 3)
for i, cm in enumerate(colormap):
    cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def image2label(im):
    data = im.astype('int32')
    # cv2.imread. default channel layout is BGR
    idx = (data[:, :, 2] * 256 + data[:, :, 1]) * 256 + data[:, :, 0]
    return np.array(cm2lbl[idx])


def dict_to_tf_example(data, label):
    with open(data, 'rb') as inf:
        encoded_data = inf.read()
    img_label = cv2.imread(label)
    img_mask = image2label(img_label)
    encoded_label = img_mask.astype(np.uint8).tobytes()

    path_data, filename_data = os.path.split(data)

    height, width = img_label.shape[0], img_label.shape[1]
    if height < vgg_16.default_image_size or width < vgg_16.default_image_size:
        # 保证最后随机裁剪的尺寸
        return None

    # Your code here, fill the dict

    feature_dict = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            filename_data.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_data),
        'image/label': dataset_util.bytes_feature(encoded_label),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def create_tf_record(output_filename, file_pars):
    # Your code here

    writer = tf.python_io.TFRecordWriter(output_filename)
    for (data, label) in file_pars:
        print(data, label)
        if not os.path.exists(str(data)):
            print('Could not find ', data, ', ignoring data.')
            continue
        if not os.path.exists(str(label)):
            logging.warning('Could not find ', label, ', ignoring data.')
            continue
        try:
            tf_example = dict_to_tf_example(data, label)
            writer.write(tf_example.SerializeToString())
        except:
            logging.warning('Invalid example: ', data, ' and ', label, ', ignoring.')
    writer.close()


def read_images_names(root, train=True):
    txt_fname = os.path.join(root, 'ImageSets/Segmentation/', 'train.txt' if train else 'val.txt')

    with open(txt_fname, 'r') as f:
        images = f.read().split()

    data = []
    label = []
    counter = 0
    countOfItems = len(images) // 500
    for fname in images:
        data.append('%s/JPEGImages/%s.jpg' % (root, fname))
        label.append('%s/SegmentationClass/%s.png' % (root, fname))
        files = zip(data, label)
        if counter % 500 == 0 and counter > 0:
            # output_path = os.path.join(FLAGS.output_dir,
            output_path = os.path.join(output_dir,
                                       'fcn_train_%05dof%05d.record' % (
                                       counter // 500 - 1, countOfItems) if train else 'fcn_val_%05dof%05d.record' % (
                                       counter // 500 - 1, countOfItems))
            create_tf_record(output_path, files)
            data = []
            label = []
        counter += 1
        if counter == len(images):
            # output_path = os.path.join(FLAGS.output_dir,
            output_path = os.path.join(output_dir,
                                       'fcn_train_%05dof%05d.record' % (
                                       counter // 500, countOfItems) if train else 'fcn_val_%05dof%05d.record' % (
                                       counter // 500, countOfItems))
            create_tf_record(output_path, files)


def main(_):
    logging.info('Prepare dataset file names')
    # read_images_names(FLAGS.data_dir, True)
    # read_images_names(FLAGS.data_dir, False)
    read_images_names(data_dir, True)
    read_images_names(data_dir, False)


if __name__ == '__main__':
    tf.app.run()
