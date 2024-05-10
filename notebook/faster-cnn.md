# 目标检测

## two-stage: Faster-Rcnn Mask-Rcnn 系列

精准，速度慢

预选框（Proposal）

特征抽取(feature extraction)，proposal提取，bounding box regression(rect refine)，classification

![img](https://pic3.zhimg.com/80/v2-c0172be282021a1029f7b72b51079ffe_720w.webp)

使用RPN生成检测框

OpenCV adaboost使用滑动窗口+图像金字塔生成检测框

R-CNN使用SS(Selective Search)方法生成检测框

共享卷积层加速目标检测得到特征图

![img](https://pic1.zhimg.com/80/v2-c93db71cc8f4f4fd8cfb4ef2e2cef4f4_720w.webp)

分类层positive or negative，回归层（x， y， h， w）

损失计算

![image-20240120193155861](C:\Users\阿七\AppData\Roaming\Typora\typora-user-images\image-20240120193155861.png)

训练方法

	1. Alternating training（交替训练）
	1. Approximate joint training（联合训练）速度快，无法求取坐标偏导
	1. Non-approximate joint training

