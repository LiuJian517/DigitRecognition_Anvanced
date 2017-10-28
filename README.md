# 文件说明
* Digit_Recognition: 文字识别的主要算法
* Digit_Recognition_Class: 加入面向对象的设计模式
* model_fit.py 训练模型，并保存模型
* read_image.py 读入图片，保存为像素矩阵，用模型预测输出

* train.csv: 训练数据集
* test.csv: 测试集
* sample_submission.csv: 测试结果保存文件样板
* result.csv: 测试结果保存文件

## 模型使用
### 先用model.fit训练模型，通过sklearn包的joblib模块保存为.m文件
### read_image读入图片数据，加载模型，预测输出
