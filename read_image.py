# For example, to create an image from binary data returned by a request, you can use the following code:
'''
 from PIL import Image
 from io import BytesIO

i = Image.open(BytesIO(r.content))

'''

from PIL import Image
import numpy as np
import pickle
from sklearn.externals import joblib
import matplotlib.pyplot as plt

def ImageToMatrix(filename):
    # 读取图片
    im = Image.open(filename)
    # 显示图片
#     im.show()
    width,height = im.size
    im = im.convert("L")   # 转换成灰度图
    '''
    print(im.format, im.size, im.mode)
    format属性定义了图像的格式，如果图像不是从文件打开的，那么该属性值为None；
    size属性是一个tuple，表示图像的宽和高（单位为像素）；
    mode属性为表示图像的模式，常用的模式为：L为灰度图，RGB为真彩色，CMYK为pre-press图像。
    '''
    data = im.getdata()
    data = np.matrix(data,dtype='float')/255.0
    new_data = np.reshape(data,(width,height))
    return new_data
#     new_im = Image.fromarray(new_data)
#     # 显示图片
#     new_im.show()

def MatrixToImage(data):
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im

'''
f = open("test.csv",'r')
a = f.readline()
a = f.readline()
data = np.matrix(a)
data = np.reshape(data,(28,28))
print(data.shape)

im = Image.fromarray(data.astype(np.uint8))
im.show()
im.save("digitFromTest_2.bmp","bmp")
'''

'''
im2 = Image.open("ren.jpg")
im2.thumbnail((28,28))
out = im2.resize((28, 28))
im2.save("ren2.jpg","jpeg")
out.save("ren3.jpg",'jpeg')

'''

from Digit_Recognition import toInt,nomalizing

if __name__ == '__main__':
    # handWritingClassTest()
    # test()
    testData = []
    im = Image.open("digitFromTest_2.bmp")
    width, height = im.size
    im = im.convert("L")  # 转换成灰度图
    im_new = im.resize((28, 28))
    data = im_new.getdata()
    # data = np.matrix(data, dtype='float') / 255.0
    # new_data = np.reshape(data, (width, height))
    testData.append(data)
    '''
    for i in range(5,10):
        try:
            im = Image.open(str(i)+".bmp")
            width, height = im.size
            im = im.convert("L")  # 转换成灰度图
            im_new = im.resize((28,28))
            data = im_new.getdata()
            # data = np.matrix(data, dtype='float') / 255.0
            # new_data = np.reshape(data, (width, height))
            testData.append(data)
        except:
            pass
    '''

    model = joblib.load("svm_model.m")

    pridicted = model.predict(nomalizing(toInt(testData)))
    print(pridicted) # 成功了，输出2






