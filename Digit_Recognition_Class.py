'''
@author:LiuJian
'''

# !/usr/bin/python
# -*- coding:utf-8 -*-

from numpy import *
import operator
import csv
from datetime import datetime

class DigitRecognition():

    def __init__(self,trainFile,testFile,outputFile):
        self._trainFile = trainFile
        self._outputFile = outputFile
        self._testFile = testFile

    def loadTrainData(self):
        '''
        把测试数据读取到列表当中
        '''

        l = []
        with open(self._trainFile) as file:
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
        return self.nomalizing(self.toInt(data)), self.toInt(label)

    def toInt(self,array):
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

    def nomalizing(self,array):
        '''
        0-1化，数据规整
        '''

        m, n = shape(array)
        for i in arange(m):
            for j in arange(n):
                if array[i, j] != 0:
                    array[i, j] = 1
        return array

    def loadTestData(self):
        '''
        读取训练数据并规整
        '''

        l = []
        with open(self._testFile) as file:
            lines = csv.reader(file)
            for line in lines:
                l.append(line)

        l.remove(l[0])
        data = array(l)
        return self.nomalizing(self.toInt(data))

    def classify(self,inX, dataSet, labels, k):
        ''' 
        将数据转化为矩阵
        计算距离
        投票分类
        '''

        inX = mat(inX)
        dataSet = mat(dataSet)
        labels = mat(labels)
        dataSetSize = dataSet.shape[0]
        diffMat = tile(inX, (dataSetSize, 1)) - dataSet
        sqDiffMat = array(diffMat) ** 2
        # print(shape(sqDiffMat))
        sqDistances = sqDiffMat.sum(axis=1)
        distances = array(sqDistances) ** 0.5
        # print(shape(distances))
        sortedDistIndicies = (distances).argsort()
        classCount = {}
        # 进行投票
        for i in range(k):
            voteLabel = labels[0, sortedDistIndicies[i]]
            classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return int(sortedClassCount[0][0])

    def saveResult(self,result):
        '''
        保存结果到文件当中
        '''

        with open(self._outputFile, 'w', newline='') as myFile:
            myWriter = csv.writer(myFile)
            # 写上行头
            myWriter.writerow(["ImageId", "Label"])
            cnt = 0
            for i in result:
                cnt += 1
                tmp = []
                tmp.append(cnt)
                tmp.append(int(i))
                myWriter.writerow(tmp)

    def handWritingClassTest(self):
        '''
        调用classify()函数实现分类 
        '''

        trainData, trainLabel = self.loadTrainData()
        testData = self.loadTestData()
        # testLabel = loadTestResult()
        m, n = shape(testData)
        print(m, n)
        errorCount = 0
        resultList = []
        for i in range(m):
            # classifierResult = classify(testData[i], trainData, trainLabel, 10)
            # memory error
            classifierResult = self.classify(testData[i], trainData[0:2000], trainLabel[0:2000], 5)
            print(classifierResult)
            resultList.append(classifierResult)

        self.saveResult(resultList)

    # 利用scikit-learn包
    def pack_knn(self):
        '''
        利用scklearn库当中的SVM包，训练支持向量机分类器
        '''

        from sklearn import metrics
        from sklearn.neighbors import KNeighborsClassifier
        trainData, trainLabel = self.loadTrainData()
        print(trainData.shape, trainLabel.shape)
        testData = self.loadTestData()
        model = KNeighborsClassifier()
        model.fit(trainData, trainLabel.T)
        pridicted = model.predict(testData)
        self.saveResult(pridicted)

    def pack_svm(self):
        '''
        利用scklearn库当中的SVM包，训练支持向量机分类器
        '''

        from sklearn import svm
        trainData, trainLabel = self.loadTrainData()
        print(trainData.shape, trainLabel.shape)
        testData = self.loadTestData()
        model = svm.SVC(decision_function_shape='ovo')  # one vs one
        model.fit(trainData, trainLabel.T)
        pridicted = model.predict(testData)
        self.saveResult(pridicted)


if __name__ == '__main__':
    # handWritingClassTest()
    # test()
    result = "result_"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+".csv"
    digit_recognition = DigitRecognition("train.csv","test.csv",result)
    digit_recognition.pack_svm()




