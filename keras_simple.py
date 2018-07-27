#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
from keras.datasets import cifar10
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

np.random.seed(114514)

print("Initialized")

# 定义每次处理的图片个数
batch_size = 128

# 定义数据集中分出的类别个数，因为CIFAR10 只有10种物体
nb_classes = 10

# epoch决定训练过程的长度，越长并非总是越好，一段时间后我们会经历收益递减，根据需要调整
nb_epoch = 45

# 这里设定图片维度，已知图片为32x32
img_rows, img_cols = 32, 32

# 卷积滤波器的个数
nb_filters = 32

# 最大池化的池化面积
pool_size = (2, 2)

# 卷积核的尺寸
kernel_size = (3,3)

print("batch_size={}, nb_classes={}, nb_epoch={}, img_rows={}, img_cols={}"
    ", nb_filters={}, pool_size={}, kernel_size={}"\
    .format(batch_size,nb_classes,nb_epoch,img_rows,img_cols,nb_filters,pool_size,kernel_size))

# 加载数据
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
    input_shape  = (img_rows, img_cols, 3)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# 将类别向量转换为二进制矩阵
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size, padding='valid', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print('Compiled')

# 设置TensorBoard
tb = TensorBoard(log_dir='~/logs')

# 训练模型
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, validation_data=(X_test,Y_test), callbacks=[tb])

