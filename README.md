基于改进yolov8模型的医学智能分类系统设计与实现

采用新冠肺炎数据集，其类别共分为四类，COVID-19，LungOpacity，NORMAL，Viral Pneumonia，数据集（百度网盘链接）：链接: https://pan.baidu.com/s/1z9XLuCYRTp9oL0AXGVO1Ng 提取码: jdyt

label.py为将四种类别标注并形成文本文件，change.py文件为根据label.py形成的txt文件将其修改为yolo目录结构并自动划分数据集（验证集：测试集=8：2）

GUI.py为所设计的简单UI界面，可以选择图片进行预测，附件中为改进模型精准率所用过的注意力机制

注解：该形目仅实现对医学图像的分类，如需识别病灶具体位置需要对图片病灶位置处进行逐个标注，训练以及测试，以及改进方式相同。


