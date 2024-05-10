## YOLOv1

![img](https://pic4.zhimg.com/v2-40c8cbed60aba0fe2faa38e240b8563b_r.jpg)

![img](https://pic1.zhimg.com/v2-8630f8d3dbe3634f124eaf82f222ca94_r.jpg)

![img](https://pic3.zhimg.com/80/v2-45795a63cdbaac8c05d875dfb6fcfb5a_720w.webp)

[目标检测|YOLO原理与实现 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/32525231)

## YOLOv2

**Batch Normalization**

在YOLOv2中，每个卷积层后面都添加了Batch Normalization层，并且不再使用droput。

**High Resolution Classifier（高分辨率）**

ImageNet分类模型基本采用大小为 224×224 的图片作为输入，分辨率相对较低，不利于检测模型。所以YOLOv1在采用 224×224 分类模型预训练后，将分辨率增加至 448×448 ，并使用这个高分辨率在检测数据集上finetune。但是直接切换分辨率，检测模型可能难以快速适应高分辨率。所以YOLOv2增加了在ImageNet数据集上使用 448×448 输入来finetune分类网络这一中间过程（10 epochs）

**Convolutional With Anchor Boxes（锚框卷积）**

YOLOv1中，输入图片最终被划分为 7×7 网格，每个单元格预测2个边界框。YOLOv1最后采用的是全连接层直接对边界框进行预测，其中边界框的宽与高是相对整张图片大小的，而由于各个图片中存在不同尺度和长宽比（scales and ratios）的物体，YOLOv1在训练过程中学习适应不同物体的形状是比较困难的，这也导致YOLOv1在精确定位方面表现较差。

YOLOv2借鉴了Faster R-CNN中RPN网络的先验框（anchor boxes，prior boxes，SSD也采用了先验框）策略。YOLOv2移除了YOLOv1中的全连接层而采用了卷积和anchor boxes来预测边界框。为了使检测所用的特征图分辨率更高，移除其中的一个pool层。在检测模型中，YOLOv2不是采用 448×448 图片作为输入，而是采用 416×416 大小。因为YOLOv2模型下采样的总步长为 32 ，对于 416×416 大小的图片，最终得到的特征图大小为 13×13 ，维度是奇数，这样特征图恰好只有一个中心位置。对于一些大物体，它们中心点往往落入图片中心位置，此时使用特征图的一个中心点去预测这些物体的边界框相对容易些。所以在YOLOv2设计中要保证最终的特征图有奇数个位置。

YOLOv1，每个cell都预测2个boxes，每个boxes包含5个值： (�,�,�,ℎ,�) ，前4个值是边界框位置与大小，最后一个值是置信度（confidence scores，包含两部分：含有物体的概率以及预测框与ground truth的IOU）。但是每个cell只预测一套分类概率值（class predictions，其实是置信度下的条件概率值）,供2个boxes共享。YOLOv2使用了anchor boxes之后，每个位置的各个anchor box都单独预测一套分类概率值，这和SSD比较类似（但SSD没有预测置信度，而是把background作为一个类别来处理）。

使用anchor boxes之后，YOLOv2的mAP有稍微下降（这里下降的原因，我猜想是YOLOv2虽然使用了anchor boxes，但是依然采用YOLOv1的训练方法）。YOLOv1只能预测98个边界框（ 7×7×2 ），而YOLOv2使用anchor boxes之后可以预测上千个边界框（ 13×13×num_anchors ）。所以使用anchor boxes之后，YOLOv2的召回率大大提升，由原来的81%升至88%。

**Dimension Clusters（聚类）**

YOLOv2采用==k-means聚类方法==对训练集中的边界框做了聚类分析。因为设置先验框的主要目的是为了使得预测框与ground truth的IOU更好，所以聚类分析时选用box与聚类中心box之间的IOU值作为距离指标：
$$
d(box,centroid) = 1-IOU(box,centroid)
$$
随着聚类中心数目的增加，平均IOU值（各个边界框与聚类中心的IOU的平均值）是增加的，但是综合考虑模型复杂度和召回率，作者最终选取5个聚类中心作为先验框。

![img](https://pic3.zhimg.com/80/v2-0c6e54419d7fb2ffada1aa7ac9540c7e_720w.webp)

**New Network: Darknet-19（新的主干）**

YOLOv2采用了一个新的基础模型（特征提取器），称为Darknet-19，包括19个卷积层和5个maxpooling层，如图4所示。Darknet-19与VGG16模型设计原则是一致的，主要采用 3×3 卷积，采用 2×2 的maxpooling层之后，特征图维度降低2倍，而同时将特征图的channles增加两倍。与==NIN(Network in Network)==类似，Darknet-19最终采用==global avgpooling==做预测，并且在 3×3 卷积之间使用 1×1 卷积来压缩特征图channles以降低模型计算量和参数。Darknet-19每个卷积层后面同样使用了batch norm层以加快收敛速度，降低模型过拟合。

**Direct location prediction（绝对位置预测）**

YOLOv2借鉴RPN网络使用anchor boxes来预测边界框相对先验框的offsets。边界框的实际中心位置 ($$x$$,$$y$$) ，需要根据预测的坐标偏移值 ($$t_x$$,$$t_y$$) ，先验框的尺度 ($$w_a$$,$$h_a$$) 以及中心坐标 ($$x_a$$,$$y_a$$) （特征图每个位置的中心点）来计算：
$$
x = (t_x×w_a)-x_a\\y = (t_y×h_a)-y_a
$$
但是上面的公式是无约束的，预测的边界框很容易向任何方向偏移，因此每个位置预测的边界框可以落在图片任何位置，这导致模型的不稳定性，在训练时需要很长时间来预测出正确的offsets。

YOLOv2弃用了这种预测方式，而是沿用YOLOv1的方法，就是预测边界框中心点相对于对应cell左上角位置的相对偏移值，为了将边界框中心点约束在当前cell中，使用sigmoid函数处理偏移值，这样预测的偏移值在(0,1)范围内（每个cell的尺度看做1）。总结来看，根据边界框预测的4个offsets $$t_x$$,$$t_y$$,$$t_w$$,$$t_h$$，可以按如下公式计算出边界框实际位置和大小：
$$
b_x = \sigma(t_x)+c_x\\b_y = \sigma(t_y)+c_y\\b_w = p_we^{t_w}\\b_h = p_he^{t_h}
$$
如果再将上面的4个值分别乘以图片的宽度和长度（像素点值）就可以得到边界框的最终位置和大小了。这就是YOLOv2边界框的整个解码过程。约束了边界框的位置预测值使得模型更容易稳定训练。

![img](https://pic3.zhimg.com/80/v2-7fee941c2e347efc2a3b19702a4acd8e_720w.webp)

**Fine-Grained Features（细粒度特征）**

YOLOv2的输入图片大小为 416×416 ，经过5次maxpooling之后得到 13×13 大小的特征图，并以此特征图采用卷积做预测。 13×13 大小的特征图对检测大物体是足够了，但是对于小物体还需要更精细的特征图（Fine-Grained Features）。因此SSD使用了多尺度的特征图来分别检测不同大小的物体，前面更精细的特征图可以用来预测小物体。YOLOv2提出了一种==passthrough==层来利用更精细的特征图。YOLOv2所利用的Fine-Grained Features是 26×26 大小的特征图（最后一个maxpooling层的输入），对于Darknet-19模型来说就是大小为 26×26×512 的特征图。passthrough层与ResNet网络的shortcut类似，以前面更高分辨率的特征图为输入，然后将其连接到后面的低分辨率特征图上。前面的特征图维度是后面的特征图的2倍，passthrough层抽取前面层的每个 2×2 的局部区域，然后将其转化为channel维度，对于 26×26×512 的特征图，经passthrough层处理之后就变成了 13×13×2048 的新特征图（特征图大小降低4倍，而channles增加4倍，图6为一个实例），这样就可以与后面的 13×13×1024 特征图连接在一起形成 13×13×3072 大小的特征图，然后在此特征图基础上卷积做预测。

![img](https://pic3.zhimg.com/80/v2-c94c787a81c1216d8963f7c173c6f086_720w.webp)

另外，作者在后期的实现中借鉴了ResNet网络，不是直接对高分辨特征图处理，而是增加了一个中间卷积层，先采用64个 1×1 卷积核进行卷积，然后再进行passthrough处理，这样 26×26×512 的特征图得到 13×13×256 的特征图。

**Multi-Scale Training（多尺度训练）**

由于YOLOv2模型中只有卷积层和池化层，所以YOLOv2的输入可以不限于 416×416 大小的图片。为了增强模型的鲁棒性，YOLOv2采用了多尺度输入训练策略，具体来说就是在训练过程中每间隔一定的iterations之后改变模型的输入图片大小。由于YOLOv2的下采样总步长为32，输入图片大小选择一系列为32倍数的值： {320,352,...,608} ，输入图片最小为 320×320 ，此时对应的特征图大小为 10×10 （不是奇数了，确实有点尴尬），而输入图片最大为 608×608 ，对应的特征图大小为 19×19 。在训练过程，每隔10个iterations随机选择一种输入图片大小，然后只需要修改对最后检测层的处理就可以重新训练。采用Multi-Scale Training策略，YOLOv2可以适应不同大小的图片，并且预测出很好的结果。



YOLOv2的一大创新是采用Multi-Scale Training策略，这样同一个模型其实就可以适应多种大小的图片了。

**YOLOv2的训练**

第一阶段就是先在ImageNet分类数据集上预训练Darknet-19，此时模型输入为 224×224 ，共训练160个epochs。然后第二阶段将网络的输入调整为 448×448 ，继续在ImageNet数据集上finetune分类模型，训练10个epochs，此时分类模型的top-1准确度为76.5%，而top-5准确度为93.3%。第三个阶段就是修改Darknet-19分类模型为检测模型，并在检测数据集上继续finetune网络。网络修改包括网路结构可视化：移除最后一个卷积层、global avgpooling层以及softmax层，并且新增了三个 3×3×2014卷积层，同时增加了一个passthrough层，最后使用 1×1 卷积层输出预测结果，输出的channels数为： num_anchors×(5+num_classes) ，和训练采用的数据集有关系。

![img](https://pic4.zhimg.com/80/v2-b23fdd08f65266f7af640c1d3d00c05f_720w.webp)

![img](https://pic3.zhimg.com/80/v2-58f1eec5594de9e887f22ac590b33062_720w.webp)

尽管YOLOv2和YOLOv1计算loss处理上有不同，但都是采用均方差来计算loss。另外需要注意的一点是，在计算boxes的 $$w$$ 和 $$h$$ 误差时，YOLOv1中采用的是平方根以降低boxes的大小对误差的影响，而YOLOv2是直接计算，但是根据ground truth的大小对权重系数进行修正



最终的YOLOv2模型在速度上比YOLOv1还快（采用了计算量更少的Darknet-19模型），而且模型的准确度比YOLOv1有显著提升。

**YOLO9000**

提出了一种层级分类方法（Hierarchical classification），主要思路是根据各个类别之间的从属关系（根据WordNet）建立一种树结构WordTree，结合COCO和ImageNet建立的WordTree如下图所示：

![img](https://pic3.zhimg.com/80/v2-bda21cf48f8127c48fede60d81c4d97e_720w.webp)

WordTree中的根节点为"physical object"，每个节点的子节点都属于同一子类，可以对它们进行softmax处理。在给出某个类别的预测概率时，需要找到其所在的位置，遍历这个path，然后计算path上各个节点的概率之积。

![img](https://pic1.zhimg.com/80/v2-311fbb6f571ab8889c850671a42a94f8_720w.webp)

在训练时，如果是检测样本，按照YOLOv2的loss计算误差，而对于分类样本，只计算分类误差。在预测时，YOLOv2给出的置信度就是 Pr(physical object) ，同时会给出边界框位置以及一个树状概率图。在这个概率图中找到概率最高的路径，当达到某一个阈值时停止，就用当前节点表示预测的类别。

通过联合训练策略，YOLO9000可以快速检测出超过9000个类别的物体。

[目标检测|YOLOv2原理与实现(附YOLOv3) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/35325884)

## YOLO v3

YOLOv3最大的变化包括两点：使用残差模型和采用FPN架构（拼接底层特征图）。

YOLOv3的特征提取器是一个残差模型，因为包含53个卷积层，所以称为Darknet-53，从网络结构上看，相比Darknet-19网络使用了残差单元，所以可以构建的更深。

另外一个点是采用FPN架构（Feature Pyramid Networks for Object Detection）来实现多尺度检测。YOLOv3采用了3个尺度的特征图（当输入为416×416 时）： (13×13) ， (26×26) ， (52×52) ，VOC数据集上的YOLOv3网络结构如图15所示，其中红色部分为各个尺度特征图的检测结果。YOLOv3每个位置使用3个先验框，所以使用k-means得到9个先验框，并将其划分到3个尺度特征图上，尺度更大的特征图使用更小的先验框，和SSD类似。

![img](https://pic4.zhimg.com/v2-ffbc5b713c98c13e2659bb528b05fd67_r.jpg)

voc数据集有20个类别，最下面红框中(13，13，75)表示预测结果的shape，实际上是13,13,3×25,表示有13*13的网格，每个网格有3个先验框（又称锚框，anchors，先验框下面有解释），每个先验框有25个参数(20个类别+5个参数)，这5个参数分别是x_offset、y_offset、height、width与置信度confidence，用这3个框去试探，试探是否框中有物体，如果有，就会把这个物体给框起来。如果是基于coco的数据集就会有80种类别，最后的维度应该为3x(80+5)=255，最上面两个预测结果shape同理。最后图中有三个红框原因就是有些物体相对在图中较大，就用13×13检测，物体在图中比较小，就会归为52×52来检测。

yolov3主干网络为Darknet53，重要的是使用了残差网络Residual，darknet53的每一个卷积部分使用了特有的DarknetConv2D结构，每一次卷积的时候进行l2正则化，完成卷积后进行BatchNormalization标准化与LeakyReLU激活函数。

[【yolov3详解】一文让你读懂yolov3目标检测原理_yolov3目标检测步骤流程图-CSDN博客](https://blog.csdn.net/weixin_39615182/article/details/109752498)

## YOLO v4

现阶段的目标检测器主要由4部分组成：
**Input**、**Backbone**、**Neck**、**Head**:

![img](https://img-blog.csdnimg.cn/20200514145553469.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTU2MDQwMg==,size_1,color_FFFFFF,t_70#pic_center)

YOLOv4模型由以下部分组成：

- **CSPDarknet53**作为骨干网络BackBone；
- **SPP**作为Neck的附加模块，**PANet**作为Neck的特征融合模块；
- **YOLOv3**作为Head。

网络结构：

![img](https://img-blog.csdnimg.cn/20200609151524354.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTU2MDQwMg==,size_1,color_FFFFFF,t_70#pic_center)

1. **CSPDarknet53**

	CSPDarknet53是在Darknet53的每个大残差块上加上CSP

	![在这里插入图片描述](https://img-blog.csdnimg.cn/20200514155014399.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTU2MDQwMg==,size_16,color_FFFFFF,t_70#pic_center)

	Darknet53的结构如下图所示，共有5个大残差块，每个大残差块所包含的小残差单元个数为1、2、8、8、4。

	![在这里插入图片描述](https://img-blog.csdnimg.cn/20200514161920351.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTU2MDQwMg==,size_16,color_FFFFFF,t_70#pic_center)

	**（1）Darknet53分块1加上CSP后的结果**，对应layer 0~layer 10。其中，layer [0, 1, 5, 6, 7]与分块1完全一样，而 layer [2, 4, 8, 9, 10]属于CSP部分。

	![在这里插入图片描述](https://img-blog.csdnimg.cn/20200514160929452.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTU2MDQwMg==,size_1,color_FFFFFF,t_70#pic_center)

	后续残差块略。。。

2. **SPP（空间金字塔池化）实现**

	![在这里插入图片描述](https://img-blog.csdnimg.cn/20200514174402114.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTU2MDQwMg==,size_1,color_FFFFFF,t_70#pic_center)

	![在这里插入图片描述](https://img-blog.csdnimg.cn/20200514174150991.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTU2MDQwMg==,size_1,color_FFFFFF,t_70#pic_center)

	![在这里插入图片描述](https://img-blog.csdnimg.cn/20200514174200818.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTU2MDQwMg==,size_1,color_FFFFFF,t_70#pic_center)

	3. **PANet实现**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200514180255816.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTU2MDQwMg==,size_16,color_FFFFFF,t_70#pic_center)

[最详细的YOLOv4网络结构解析-CSDN博客](https://blog.csdn.net/weixin_41560402/article/details/106119774)

## YOLO v5

![img](https://img-blog.csdnimg.cn/20210211163722891.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1daWjE4MTkxMTcxNjYx,size_16,color_FFFFFF,t_70#pic_center)

1. 输入端

	**Mosaic数据增强** 

	Yolov5的输入端采用了和Yolov4一样的Mosaic数据增强的方式。

	**自适应锚框计算** 

	在Yolov3、Yolov4中，训练不同的数据集时，计算初始锚框的值是通过单独的程序运行的。但Yolov5中将此功能嵌入到代码中，每次训练时，自适应的计算不同训练集中的最佳锚框值。当然，如果觉得计算的锚框效果不是很好，也可以在代码中将自动计算锚框功能**关闭**。

	![img](https://pic3.zhimg.com/80/v2-807f03432ab4deb4a959d2dddd95923e_720w.webp)

	**自适应图片缩放**

	作者认为，在项目实际使用时，很多图片的长宽比不同，因此缩放填充后，两端的黑边大小都不同，而如果填充的比较多，则存在信息冗余，影响推理速度。

	因此在Yolov5的代码中datasets.py的letterbox函数中进行了修改，对原始图像**自适应的添加最少的黑边**。图像高度上两端的黑边变少了，在推理时，计算量也会减少，即目标检测速度会得到提升。（方法详见链接）

2. backbone

	**Focus结构**

	Focus结构，在Yolov3&Yolov4中并没有这个结构，其中比较关键是切片操作。切片示意图如右，4×4×3的图像切片后变成2×2×12的特征图。

	![img](https://pic3.zhimg.com/80/v2-5c6d24c95f743a31d90845a4de2e5a36_720w.webp)

	**CSP结构**

	Yolov4网络结构中，借鉴了CSPNet的设计思路，在主干网络中设计了CSP结构。

	Yolov5与Yolov4不同点在于，Yolov4中只有主干网络使用了CSP结构。

3. neck

	Yolov5现在的Neck和Yolov4中一样，都采用FPN+PAN的结构

	![img](https://pic4.zhimg.com/80/v2-f903f571b62f07ab7c30d72d54e5e0c3_720w.webp)

	Yolov4的Neck结构中，采用的都是普通的卷积操作。而Yolov5的Neck结构中，采用借鉴CSPnet设计的CSP2结构，加强网络特征融合的能力。

4. 输出端

	**Bounding box损失函数**

	Yolov5中采用其==CIOU_Loss==做Bounding box的损失函数。

	**nms非极大值抑制**

	在目标检测的后处理过程中，针对很多目标框的筛选，通常需要nms操作。

	因为CIOU_Loss中包含影响因子v，涉及groudtruth的信息，而测试推理时，是没有groundtruth的。

	所以Yolov4在DIOU_Loss的基础上采用DIOU_nms的方式，而Yolov5中采用加权nms的方式。

[深入浅出Yolo系列之Yolov5核心基础知识完整讲解 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/172121380)

## YOLO v6

