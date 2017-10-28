
from numpy import *
import operator
import csv
import pickle
from sklearn.externals import joblib


def loadTrainData():
    '''
    把测试数据读取到列表当中

    '''

    l = []
    with open('train.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)

    # 去掉第一行的描述信息
    l.remove(l[0])
    l = array(l)

    # 提取标签
    label = l[:, 0]

    # 提取数据
    data = l[:, 1:]
    return nomalizing(toInt(data)), toInt(label)


def toInt(array):
    '''
    把数据转成整数类型

    '''
    array = mat(array)
    m, n = shape(array)
    newArray = zeros((m, n))
    for i in arange(m):
        for j in arange(n):
            newArray[i, j] = int(array[i, j])
    return newArray


def nomalizing(array):
    # 0-1化，数据规整
    m, n = shape(array)
    for i in arange(m):
        for j in arange(n):
            if array[i, j] != 0:
                array[i, j] = 1
    return array


def loadTestData():
    # 读取训练数据并规整
    l = []
    with open('test.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    data = array(l)
    return nomalizing(toInt(data))

def pack_svm():
    '''
    利用scklearn库当中的SVM包，训练支持向量机分类器
    '''
    from sklearn import svm
    trainData, trainLabel = loadTrainData()
    print(trainData.shape, trainLabel.shape)
    # testData = loadTestData()
    model = svm.SVC(decision_function_shape='ovo')  # one vs one
    model.fit(trainData, trainLabel.T)
    joblib.dump(model,"svm_model.m")

pack_svm()