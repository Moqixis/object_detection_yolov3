"""YOLO_v3 Model Defined in Keras."""

from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from yolo3.utils import compose

# 定义卷积层,对conv2d函数进行了功能修改
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)} # 正则化方式改为L2正则化,默认是None（将核权重参数w进行正则化）
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same' # 当步长为2时,padding的方式才为valid。默认情况下是步长为1
    darknet_conv_kwargs.update(kwargs) 
    return Conv2D(*args, **darknet_conv_kwargs)

# 卷积层+批量标准化+激活函数，作为yolov3的⼀个基本块
def DarknetConv2D_BN_Leaky(*args, **kwargs): # *args打包成tuple，**kwargs打包成字典
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

# 定义残差块，由基本块和全零填充组
def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
                DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
        x = Add()([x,y])
    return x

# 定义主干网络darknet的结构，由基本块和残差块组成
def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x

# 定义网络的其他部分 DBL*5 + DBL + conv
def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)
    y = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D(out_filters, (1,1)))(x)
    return x, y

# 定义整个网络结构
def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    # 13*13预测图，输出的x是(?, 13, 13, 512)，输出的y是(?, 13, 13, 18)
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))

    # 26*26预测图，输出的x是(?, 26, 26, 256)，输出的y是(?, 26, 26, 18)
    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    # 52*52预测图，输出的x是(?, 52, 52, 128)，输出的y是(?, 52, 52, 18)
    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    # 整个模型的输入inputs(?, 416, 416, 3)，输出为3个尺度的预测层，即[y1, y2, y3]
    return Model(inputs, [y1,y2,y3])

# 定义简化版的⽹络结构
def tiny_yolo_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    x1 = compose(
            DarknetConv2D_BN_Leaky(16, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(32, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(64, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(128, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(256, (3,3)))(inputs)
    x2 = compose(
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(512, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'),
            DarknetConv2D_BN_Leaky(1024, (3,3)),
            DarknetConv2D_BN_Leaky(256, (1,1)))(x1)
    y1 = compose(
            DarknetConv2D_BN_Leaky(512, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    # 整个模型的输入inputs(?, 416, 416, 3)，输出为2个尺度的预测层，即[y1, y2]
    return Model(inputs, [y1,y2])

# 从输出层解码得到边框信息，返回的box_xy和box_wh是相对于特征图的位置
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2]) # [1,1,1,3,2]

    grid_shape = K.shape(feats)[1:3] # height, width
    # grid_y和grid_x(相当于公式里的c_x,c_y)，用于生成网格grid（目标中心点所在网格左上角距最左上角相差的格子数）
    # 创建y轴的0~12的组合grid_y，再创建x轴的0~12的组合grid_x，
    # 将两者拼接concatenate，就是grid，即0~12等差分布的(13,13,1,2)的张量
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats)) # K.cast():把grid中值的类型变为和feats中值的类型一样

    # (batch_size,13,13,3,18)    
    # 把yolobody的输出，转成跟y_true对应的维度数据
    # 将feats的最后一维展开，方便将anchors与其他数据（类别数+4个框值+框置信度）分离   
    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    # 将feats中x，y相关的信息sigmoid后(将feats映射到(0,1)),与所在网格位置(偏移)加和，再除以grid_shape(归一化)
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats)) # 实际上就是除以13或26或52
    # 将feats中w，h的值，经过exp正值化，再乘以anchors_tensor的anchor box，再除以图片宽高（归一化）
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    # 置信度和类别
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])
    # ...操作符，在Python中，“...”(ellipsis)操作符，表示其他维度不变，只操作最前或最后1维

    # 在计算loss的时候返回如下参数
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    # box_xy对应框的中心点, box_wh对应框的宽和高
    return box_xy, box_wh, box_confidence, box_class_probs

# box相对于整张图片的中心坐标转换成box的左上角 右下角的坐标
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    # input_shape为输入图像的尺寸，image_shape为（416*416）
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    # 取 min(w/img_w, h/img_h)这个比例来缩放
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    # 计算缩放图相对于原图的偏离量，除input_shape是做归一化，因为输入的box_xy,box_wh都是做了归一
    offset = (input_shape-new_shape)/2./input_shape
    # 根据缩放比例计算边界框的长和宽
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes

# 获得box与得分
def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4]) # （?,13,13,3,4）->(?,4)  ?:框的数目
    box_scores = box_confidence * box_class_probs # 计算锚框的得分
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores

# 删除冗余框，保留最优框，用到NMS非极大值抑制算法
def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,          # placeholder类型的TF参数，默认(416, 416)
              max_boxes=20,         # 每张图每类最多检测到20个框同类别框的IoU阈值，大于阈值的重叠框被删除
              score_threshold=.6,   # 框置信度阈值，小于阈值的框被删除，需要的框较多，则调低阈值，需要的框较少，则调高阈值
              iou_threshold=.5):    # 同类别框的IoU阈值，大于阈值的重叠框被删除，重叠物体较多，则调高阈值，重叠物体较少，则调低阈值
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs) # yolo的输出层数；num_layers = 3  -> 13-26-52
    # 每层分配3个anchor box
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] # default setting
    # 输入shape(?,13,13,255);即第一维和第二维分别*32  ->13*32=416; input_shape:(416,416)
    # yolo_outputs=[(batch_size，13,13,255)，(batch_size，26,26,255)，(batch_size，52,52,255)]
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32 # input_shape=416*416
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0) # K.concatenate:将数据展平 ->(?,4)

    # 经过（1）阈值的删选，（2）非极大值抑制的删选
    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    # 对每一个类进行判断
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c]) 
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c]) # 将第c类中得分大于阈值的坐标挑选出来
        # 非极大抑制，去掉box重合程度高的那一些
        """原理: (1)从最大概率矩形框F开始，分别判断A~E与F的重叠度IOU是否大于某个设定的阈值;
                 (2)假设B、D与F的重叠度超过阈值，那么就扔掉B、D；并标记第一个矩形框F，是我们保留下来的。
                 (3)从剩下的矩形框A、C、E中，选择概率最大的E，然后判断E与A、C的重叠度，
                    重叠度大于一定的阈值，那么就扔掉；并标记E是我们保留下来的第二个矩形框。
                就这样一直重复，找到所有被保留下来的矩形框。"""
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c # 将class_box_scores中的数变成1
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_

# 将真实框的位置预处理为训练输入格式，以便与预测值比较
def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5) 检测框，批次数，box框数，每个框5个值
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32 图片尺寸（高宽）
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    # true_boxes包含5个参数，分别是x_min、y_min、x_max、y_max、class_id
    # 检查有无异常数据 即txt提供的box id 是否存在大于 num_class的情况
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)//3 # default setting
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    # 取框的真实值，获取其框的中心及其宽高
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]   
    # 中心坐标 和 宽高 都变成 相对于input_shape的比例
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    # m是batch size 即是输入图片的数量
    m = true_boxes.shape[0]
    # grid_shape [13,13]  [26,26]  [52,52]
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    # y_true是全0矩阵（np.zeros）列表，即[(m,13,13,3,6), (m,26,26,3,6), (m,52,52,3,6)]，其中6=1+4+1
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting. 在原先axis出添加一个维度,由(9,2)转为(1,9,2)
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.     # 网格中心为原点,即网格中心坐标为(0,0),计算出anchor右下角坐标
    anchor_mins = -anchor_maxes     # 计算出左上标
    valid_mask = boxes_wh[..., 0]>0 # 取有效框

    # 对每一张图片进行处理
    for b in range(m):
        # Discard zero rows. 如果没有box则检测下一张图片
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        # Expand dim to apply broadcasting.  (T, 1, 2)
        wh = np.expand_dims(wh, -2)
        # 将真实框与先验框进行对比运算，计算IoU
        box_maxes = wh / 2.     # 假设　bouding box 的中心也位于网格的中心
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)  # 逐位比较
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1] # 宽*高
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # 计算每一幅图中，真实框与那个先验框最匹配
        # shape为(T)，代表每一个框最匹配的先验框的位置0123456789
        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor): # t是box的序号, n是最优anchor的序号
            for l in range(num_layers):     # l是层号
                if n in anchor_mask[l]:     # 如果最优anchor在层l中,则设置其中的值,否则默认为0
                    # 计算该目标在第l个特征层所处网格的位置
                    # true_boxes[b, t, 0]，其中b是批次序号、t是box序号，第0位是x，第1位是y；
                    # grid_shapes是3个检测图的尺寸，将归一化的值，与框长宽相乘，恢复为具体值；
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    # 找到best_anchor索引的索引
                    k = anchor_mask[l].index(n) # k是在anchor box中的序号
                    c = true_boxes[b,t, 4].astype('int32') # c是类别，true_boxes的第4位
                    # 保存到y_true中
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4] # 将xy和wh放入y_true中
                    y_true[l][b, j, i, k, 4] = 1    # 将y_true的第4位框的置信度设为1
                    y_true[l][b, j, i, k, 5+c] = 1  # 将y_true第5~n位的类别设为1

    return y_true

# 计算两个矩形框的交并比
def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # 得到左上角点，右下角点坐标，然后计算
    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou

# 损失函数
def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    """
    true_boxes : 实际框的位置和类别，我们的输入。三个维度：
    第一个维度：图片张数
    第二个维度：一张图片中有几个实际框
    第三个维度： [x, y, w, h, class]，x,y,w,h 均是除以图片分辨率得到的[0,1]范围的值。
    anchors : 实际anchor boxes 的值，论文中使用了五个。[w,h]，都是相对于gird cell 长宽的比值。二个维度：
    第一个维度：anchor boxes的数量，这里是5
    第二个维度：[w,h]，w,h,都是相对于gird cell 长宽的比值。
    """
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    num_layers = len(anchors)//3 # default setting
    # 将预测结果和实际ground truth分开
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    # 先验框分组
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
    # 得到input_shpae为416,416  13*13 的宽高回归416*416 
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    # 得到网格的shape为13,13;26,26;52,52
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    # 模型的batch size
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor 输入模型的图片总量
    mf = K.cast(m, K.dtype(yolo_outputs[0])) # 调整类型

    # 循环计算每1层的损失值，累加到一起
    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]       # 置信度信息
        true_class_probs = y_true[l][..., 5:]   # 类别信息

        # 通过最后一层的输出，构建pred_box。yolo_head将最后一层的特征转换为b-box的信息
        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # 生成真实数据
        # Darknet raw box to calculate loss.
        # raw_true_xy：在网格中的中心点xy，偏移数据，值的范围是0~1 （相对于一个小格子而言）
        # raw_true_wh：在网络中的wh相对于anchors的比例，再转换为log形式，范围是有正有负
        # y_true的第0和1位是中心点xy的相对于规范化图片的位置，第2和3位是宽高wh的相对于规范化图片的位置，范围都是0~1
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
        # box_loss_scale：计算wh权重，取值范围(1~2)
        # w*h越小，面积越小，在和anchor做比较的时候，iou必然就小，导致"存在物体"的置信度就越小。也就是object_mask越小
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4] # 2-box_ares 避免大框的误差对loss 比小框误差对loss影响大

        # 根据置信度生成二值向量
        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True) 
        object_mask_bool = K.cast(object_mask, 'bool')
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0]) # 挑选出置信度大于0的框的相应的坐标
            iou = box_iou(pred_box[b], true_box) # pre_box是通过yolo_head解码之后的xywh
            best_iou = K.max(iou, axis=-1)
            # 根据IoU忽略阈值生成ignore_mask，抑制IoU小于最大阈值的预测框
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask]) # 传入loop_body函数初值为b=0，ignore_mask
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1) # 扩展维度用于计算loss

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
            (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
    return loss
