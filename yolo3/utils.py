"""Miscellaneous utility functions."""

from functools import reduce

from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

# 组合多个函数，上一个函数的输出作为下一个函数的输入
def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

# 等⽐例缩放图片
def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

# 返回[a,b)之间的随机数
def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

# 数据预处理，对图像属性做⼀些变换，模拟实时图像数据,并且把RGB值归⼀化
def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    #  annotation_line: 数据集中的某一行对应的图片
    #  input_shape:  yolo网络输入图片的大小416*416
    #  jitter:控制图片的宽高的扭曲比率,jitter=.5表示在0.5到1.5之间进行扭曲
    #  hue: 代表hsv色域中三个通道中的色调进行扭曲，色调（H）=.1
    #  sat: 代表hsv色域中三个通道中的饱和度进行扭曲，饱和度(S)=1.5
    #  val: 代表hsv色域中三个通道中的明度进行扭曲，明度（V）=1.5
    #  return:
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    # 对该行的图片中的目标框进行一个划分
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])  

    if not random:
        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data=0
        if proc_img:
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)/255.

        # correct boxes
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            if len(box)>max_boxes: box = box[:max_boxes]
            box[:, [0,2]] = box[:, [0,2]]*scale + dx
            box[:, [1,3]] = box[:, [1,3]]*scale + dy
            box_data[:len(box)] = box

        return image_data, box_data

    # 对图像进行缩放并且进行长和宽的扭曲
    # resize image
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    # rand(.25, 1)会把原始的图片进行缩小，图片的边缘加上灰条，可以训练网络对小目标的检测能力。rand(1,2)放大图像
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw,nh), Image.BICUBIC)

    # 将图像多余的部分加上灰条，保证图片的大小为w,h = 416,416
    # place image
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # 翻转图像
    # flip image or not
    flip = rand()<.5 # 有50%的几率发生翻转
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT) # 左右翻转

    # 色域扭曲
    # 色域扭曲是发生在hsv色域上，hsv色域是有色调H、饱和度S、明度V三者控制，调整这3个值调整色域扭曲的比率
    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = rgb_to_hsv(np.array(image)/255.) # 将图片从RGB图像调整到hsv色域上之后，再对其色域进行扭曲
    x[..., 0] += hue
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x>1] = 1
    x[x<0] = 0
    image_data = hsv_to_rgb(x) # numpy array, 0 to 1

    # 将box进行调整，对原图片进项扭曲后，也要对原图片中的框也进行相应的调整
    # correct boxes
    box_data = np.zeros((max_boxes,5))
    if len(box)>0:
        np.random.shuffle(box)
        # 扭曲调整
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
        # 旋转调整
        if flip: box[:, [0,2]] = w - box[:, [2,0]]
        # 因为调整后不再图像中的目标框的调整
        box[:, 0:2][box[:, 0:2]<0] = 0
        box[:, 2][box[:, 2]>w] = w
        box[:, 3][box[:, 3]>h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
        if len(box)>max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data
