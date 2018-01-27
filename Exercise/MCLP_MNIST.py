# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import struct
import os
from PIL import Image
from sklearn.neural_network import MLPClassifier

def loadMNISTImages(filename):
    #判断文件是否存在
    if not os.path.exists(filename):
        print(filename + 'is not exist!')
    else:
        #建立对象
        fileObject = open(filename, 'rb')
        #读入所有数据
        bufferData = fileObject.read()
        
        #读取前四个数据
        headData = struct.unpack_from('>IIII', bufferData, 0)
        #定位数据的起始位置
        dataOffset = struct.calcsize('>IIII')
        
        #获取数据的属性
        ImagesMagic = headData[0]
        ImagesNum = headData[1]
        ImagesRow = headData[2]
        ImagesCol = headData[3]
        
        #打印数据属性
#        print("Images magic: ", ImagesMagic)
#        print("Images Number: ", ImagesNum)
#        print("Images Row: ", ImagesRow)
#        print("Images Col: ", ImagesCol)
        
        dataBits = ImagesNum*ImagesRow*ImagesCol
        
        dataFormat = '>' + str(dataBits) + 'B'
        
        ImagesData = struct.unpack_from(dataFormat, bufferData, dataOffset)
        
        #关闭文件
        fileObject.close()
        
        #重构训练矩阵
        ImagesMatrix = np.reshape(ImagesData, [ImagesNum, ImagesCol*ImagesRow])
        
        return ImagesMatrix
    

def loadMNISTLabels(filename):
    if not os.path.exists(filename):
        print(filename + 'is not exist!')
    else:
        #建立文件对象
        fileObject = open(filename, 'rb')
        #读取所有数据
        bufferData = fileObject.read()
        
        #获取数据的头
        headData = struct.unpack_from('>II', bufferData, 0)
        #定位数据起始位置
        dataOffset = struct.calcsize('>II')
        
        #获取数据信息
        LabelsMagic = headData[0]
        LabelsNum = headData[1]
        
        dataFormat = '>' + str(LabelsNum) + 'B'
        
        LabelsData = struct.unpack_from(dataFormat, bufferData, dataOffset)
        
        fileObject.close()
        
        #重构数据矩阵
        LabelsMatrix = np.reshape(LabelsData, [LabelsNum])
        return LabelsMatrix
        

trainData = loadMNISTImages('C:/Users/clzhs/Desktop/Datasets/mnist/mnist/train-images.idx3-ubyte')
trainlabel = loadMNISTLabels('C:/Users/clzhs/Desktop/Datasets/mnist/mnist/train-labels.idx1-ubyte')
testData = loadMNISTImages('C:/Users/clzhs/Desktop/Datasets/mnist/mnist/t10k-images.idx3-ubyte')
testLabel = loadMNISTLabels('C:/Users/clzhs/Desktop/Datasets/mnist/mnist/t10k-labels.idx1-ubyte')

trainData = trainData / 255
testData = testData / 255

#构建多层感知器
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter= 500, alpha=1e-4, solver= 'sgd', verbose=10, tol=1e-4, random_state=1, learning_rate_init=0.001)
mlp.fit(trainData, trainlabel)

print("Training Datasets Scores: ", mlp.score(trainData, trainlabel))
print("Testing Datasets Scores: ", mlp.score(testData, testLabel))