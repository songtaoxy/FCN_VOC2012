#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os

import cv2
import numpy as np
import tensorflow as tf
from vgg import vgg_16

'''
功能:将数据集转成tensorflo的tfrecord格式
直接生成一个文件:

fcn_train.record
fcn_val.record

注意: 该文件必须在命令行下执行才能生效, 如果在IDE, 比如 pycharm上执行会出错

cd到该文件所在目录
python convert_fcn_dataset_v1_cmd.py

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

data_dir = '../datasets/raw/VOC2012/'
output_dir = '../datasets/train/'


# FLAGS = flags.FLAGS

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


# 每个像素点有 0 ~ 255 的选择，RGB 三个通道
cm2lbl = np.zeros(256**3)

for i, cm in enumerate(colormap):
    cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def image2label(im):

    # data = im.astype('int32').asnumpy()这么写会报错:
    # AttributeError: 'numpy.ndarray' object has no attribute 'asnumpy'
    data = im.astype('int32')
    # 用opencv读入的图片按照BRG
    idx = (data[:,:,2]*256+data[:,:,1])*256+data[:,:,0]

    return np.array(cm2lbl[idx])

# 将object_detection.utils.dataset_util下面几个定义TFRecord数据集的工具函数拿过来:
def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def dict_to_tf_example(data, label):
    with open(data, 'rb') as inf:
        encoded_data = inf.read()
    img_label = cv2.imread(label)
    img_mask = image2label(img_label)
    encoded_label = img_mask.astype(np.uint8).tobytes()

    height, width = img_label.shape[0], img_label.shape[1]
    if height < vgg_16.default_image_size or width < vgg_16.default_image_size:
        # 保证最后随机裁剪的尺寸
        # 小于要求大小的图片全部过滤掉,即 H & W 要大于等于 224
        return None

    # Your code here, fill the dict
    # 提取文件名
    image_name=os.path.splitext(os.path.basename(data))[0]

    # 文件名要转化为b'xxx',否则会报错
    # TypeError: 'xxx' has type str, but expected one of: bytes
    image_name=image_name.encode()

    # 在sample-code文件夹中所有文件搜索image/
    # 发现dataset.py中出现了调用:
    # image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    # label = tf.decode_raw(features['image/label'], tf.uint8)
    # 这样就了解了字典应该怎样补充了.

    feature_dict = {
        'image/height': int64_feature(height), # 标签图片高,训练无用的feature
        'image/width': int64_feature(width), # 标签图片宽,训练无用的feature
        'image/filename': bytes_feature(image_name), # 图片名,注意要encode,训练无用的feature
        'image/encoded':  bytes_feature(encoded_data), # 训练图片
        'image/label': bytes_feature(encoded_label), # 标签图片
        'image/format':bytes_feature('jpeg'.encode('utf8')), # 训练无用的feature
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example

def create_tf_record(output_filename, file_pars):

    """Creates a TFRecord file from examples.
    Args:
        output_filename: tfrecord的保存路径+文件名.
        file_pars: 原图像和标签图像轮机的元组列表.
        列表元素: ('.../JPEGImages/xxx.jpg' '.../SegmentationClass/xxx.png')
    """
    writer = tf.python_io.TFRecordWriter(output_filename)

    for img_path, label_path in file_pars:
        # print(img_path,label_path)
        tf_example = dict_to_tf_example(img_path, label_path)

        # 只有非none的返回才进行输出,否则
        # AttributeError: 'NoneType' object has no attribute 'SerializeToString'
        if not(tf_example is None):
            writer.write(tf_example.SerializeToString())

    writer.close()

def read_images_names(root, train=True):
    txt_fname = os.path.join(root, 'ImageSets/Segmentation/', 'train.txt' if train else 'val.txt')

    with open(txt_fname, 'r') as f:
        images = f.read().split()

    data = []
    label = []
    for fname in images:
        data.append('%s/JPEGImages/%s.jpg' % (root, fname))
        label.append('%s/SegmentationClass/%s.png' % (root, fname))
    return zip(data, label)


def main(_):
    logging.info('Prepare dataset file names')

    '''
    注释掉的是 命令行脚本执行的方法
    当前使用的是本地直接运行
    '''
    # train_output_path = os.path.join(FLAGS.output_dir, 'fcn_train.record')
    # val_output_path = os.path.join(FLAGS.output_dir, 'fcn_val.record')

    train_output_path = os.path.join(output_dir, 'fcn_train.record')
    val_output_path = os.path.join(output_dir, 'fcn_val.record')

    # train_files = read_images_names(FLAGS.data_dir, True)
    # val_files = read_images_names(FLAGS.data_dir, False)

    train_files = read_images_names(data_dir, True)
    val_files = read_images_names(data_dir, False)

    create_tf_record(train_output_path, train_files)
    create_tf_record(val_output_path, val_files)


if __name__ == '__main__':
    tf.app.run()

#python convert_fcn_dataset_v1_sh.py --data_dir=./VOC2012 --output_dir=./
