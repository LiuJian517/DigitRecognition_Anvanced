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
model_fit.py:支持向量机分类器
model_fit_knn.py：KNN分类器

运行步骤：
1）运行model_fit.py，训练支持向量机分类器，保存结果
2）运行model_fit_knn.py,训练KNN分类器，保存结果
3） 运行digit_gen.py ,从测试集当中反向生成图片，保存在digits_gen目录中
4） 将自己手写的图片（PS、美图等）保存在number文件夹中
5） 运行read_image.py，测试图片识别的效果
6） 结果：测试集图片识别成功率：90%
	  PS生成的图片识别成功率：70%
