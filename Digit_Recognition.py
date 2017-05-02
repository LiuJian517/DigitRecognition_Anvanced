'''
@author:LiuJian
Date:2017-04-01
'''

#!/usr/bin/python
#-*- coding:utf-8 -*-
from numpy import *
import operator
import csv

def loadTrainData():
    l = []
    with open('train.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)

    l.remove(l[0])
    l = array(l)
    label = l[:,0]
    data = l[:,1:]
    return nomalizing(toInt(data)),toInt(label)

def toInt(array):
    array = mat(array)
    m,n = shape(array)
    newArray = zeros((m,n))
    for i in arange(m):
        for j in arange(n):
            newArray[i,j] = int(array[i,j])
    return newArray

def nomalizing(array):
    m,n = shape(array)
    for i in arange(m):
        for j in arange(n):
            if array[i,j] != 0:
                array[i,j] = 1
    return array

def loadTestData():
    l = []
    with open('test.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    data = array(l)
    return nomalizing(toInt(data))

def classify(inX,dataSet,labels,k):
    inX = mat(inX)
    dataSet = mat(dataSet)
    labels = mat(labels)
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = array(diffMat)**2
    # print(shape(sqDiffMat))
    sqDistances = sqDiffMat.sum(axis=1)
    distances = array(sqDistances)**0.5
    # print(shape(distances))
    sortedDistIndicies = (distances).argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[0,sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0)+1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse=True)
    return int(sortedClassCount[0][0])

def saveResult(result):
    with open('result3.csv','w',newline='') as myFile:
        myWriter = csv.writer(myFile)
        myWriter.writerow(["ImageId","Label"])
        cnt = 0
        for i in result:
            cnt+=1
            tmp = []
            tmp.append(cnt)
            tmp.append(i)
            myWriter.writerow(tmp)

def handWritingClassTest():
    trainData,trainLabel = loadTrainData()
    testData = loadTestData()
    # testLabel = loadTestResult()
    m,n = shape(testData)
    print(m,n)
    errorCount = 0
    resultList = []
    for i in range(m):
        #classifierResult = classify(testData[i], trainData, trainLabel, 10) memory error
        classifierResult = classify(testData[i],trainData[0:2000],trainLabel[0:2000],5)
        print(classifierResult)
        resultList.append(classifierResult)

    saveResult(resultList)


# 利用scikit-learn包
def test():
    from sklearn import metrics
    from sklearn.neighbors import KNeighborsClassifier
    trainData, trainLabel = loadTrainData()
    print(trainData.shape,trainLabel.shape)
    testData = loadTestData()
    model = KNeighborsClassifier()
    model.fit(trainData,trainLabel.T)
    pridicted = model.predict(testData)
    saveResult(pridicted)

def svm():
    from sklearn import svm
    trainData, trainLabel = loadTrainData()
    print(trainData.shape, trainLabel.shape)
    testData = loadTestData()
    model = svm.SVC(decision_function_shape='ovo') # one vs one
    model.fit(trainData, trainLabel.T)
    pridicted = model.predict(testData)
    saveResult(pridicted)


if __name__=='__main__':
    # handWritingClassTest()
    # test()
    svm()




