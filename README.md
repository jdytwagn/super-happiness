基于改进yolov8模型的医学智能分类系统设计与实现

本项目通过优化yolov8深度学习模型，构建一个高精度的医学分类图像系统。首先通过搜集相关数据集，并对数据集进行标注，将其分为训练集和测试集，之后对模型进行训练，得到具有较好拟合能力的模型，进而对模型进行测试，观察其在测试集中的泛化能力，通过优化yolov8的结构，进而提高模型精度。以提高其在医学图像分类中的精度。

采用新冠肺炎数据集，其类别共分为四类，COVID-19，LungOpacity，NORMAL，Viral Pneumonia，数据集（百度网盘链接）：链接: https://pan.baidu.com/s/1z9XLuCYRTp9oL0AXGVO1Ng 提取码: jdyt

label.py为将四种类别标注并形成文本文件，change.py文件为根据label.py形成的txt文件将其修改为yolo目录结构并自动划分数据集（验证集：测试集=8：2）

GUI.py为所设计的简单UI界面，可以选择图片进行预测，附件中为改进模型精准率所用过的注意力机制（其也可通过修改backbone,损失函数等进行改进）。

![image](https://github.com/user-attachments/assets/6ca3cfb1-813a-4720-8832-51ba98d8a645)

注解：该形目仅实现对医学图像的分类，如需识别病灶具体位置需要对图片病灶位置处进行逐个标注，训练以及测试，以及改进方式相同。


