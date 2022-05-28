# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

class YOLO(object):
    _defaults = {
        "model_path": 'logs/ep027-loss9.739-val_loss9.585.h5',
        "anchors_path": 'model_data/tiny_yolo_anchors.txt',
        "classes_path": 'model_data/voc_classes.txt',
        "score" : 0.3,  # 置信度阈值
        "iou" : 0.45,   # 交并比阈值
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs): # 传入参数
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    # 读取所有的类别
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # 读取所有的anchor
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    # 读入训练好的模型,如果失败则创建模型
    def generate(self):
        """①加载权重参数文件，生成检测框，得分，以及对应类别

          ②利用model.py中的yolo_eval函数生成检测框，得分，所属类别

          ③初始化时调用generate函数生成图片的检测框，得分，所属类别（self.boxes, self.scores, self.classes）"""
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            # 检查一下模型中的参数和设置的anchor,num_classes是否一样
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 生成绘制边框的颜色。
        # h: x/len(self.class_names)  s: 1.0  v: 1.0  , 乘255变成RGB
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # 为过滤的边界框生成输出张量目标 yolo_eval
        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    # 检测图片
    def detect_image(self, image):
        """开始计时
        ->①调用letterbox_image函数，即：先生成一个用“绝对灰”R128-G128-B128填充的416×416新图片，
        然后用按比例缩放（采样方式：BICUBIC）后的输入图片粘贴，粘贴不到的部分保留为灰色。
        ②model_image_size定义的宽和高必须是32的倍数；若没有定义model_image_size，
        将输入的尺寸调整为32的倍数，并调用letterbox_image函数进行缩放。
        ③将缩放后的图片数值除以255，做归一化。
        ④将（416,416,3）数组调整为（1,416,416,3）元祖，满足网络输入的张量格式：image_data。

        ->①运行self.sess.run（）输入参数：输入图片416×416，学习模式0测试/1训练。
        self.yolo_model.input: image_data，self.input_image_shape: [image.size[1], image.size[0]]，
        K.learning_phase(): 0。
        ②self.generate（），读取：model路径、anchor box、coco类别、加载模型yolo.h5.，
        对于80中coco目标，确定每一种目标框的绘制颜色，即：将（x/80,1.0,1.0）的颜色转换为RGB格式，
        并随机调整颜色一遍肉眼识别，其中：一个1.0表示饱和度，一个1.0表示亮度。
        ③若GPU>2调用multi_gpu_model()

         ->①yolo_eval(self.yolo_model.output),max_boxes=20,每张图没类最多检测20个框。
         ②将anchor_box分为3组，分别分配给三个尺度，yolo_model输出的feature map
         ③特征图越小，感受野越大，对大目标越敏感，选大的anchor box
         ->分别对三个feature map运行out_boxes, out_scores, out_classes，返回boxes、scores、classes。
         """
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else: # 按比例缩放
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.   # 归一化
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data, # 图像数据
                self.input_image_shape: [image.size[1], image.size[0]], # 图像尺寸416x416
                K.learning_phase(): 0 # 学习模式 0：测试模型。 1：训练模式
            }) # 目的为了求boxes,scores,classes，具体计算方式定义在generate（）函数内

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        # 设置字体
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300  # 设置目标框线条的宽度

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i] # 置信度

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)  # 创建一个可以在给定图像上绘图的对象
            label_size = draw.textsize(label, font) # 标签文字,返回label的宽和高（多少个pixels）

            top, left, bottom, right = box
            # 目标框的上、左两个坐标小数点后一位四舍五入,防止检测框溢出
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            # 目标框的下、右两个坐标小数点后一位四舍五入，与图片的尺寸相比，取最小值
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            # 确定标签（label）起始点位置：标签的左、下
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # 画目标框，线条宽度为thickness
            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            #文字背景
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return image

    def close_session(self):
        self.sess.close()

# 检测视频,分帧调⽤detect_image()
# def detect_video(yolo, video_path, output_path=""):
def detect_video(yolo, video_path, output_path):
    import cv2
    vid = cv2.VideoCapture(video_path) # 0是摄像头
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read() # 读入一帧图片
        if(return_value==False): # opencv读取视频，最后帧是为空，我们需要做一个判断，如果为空就跳出循环
            print("******************************************************") 
            break
        image = Image.fromarray(frame)
        image = yolo.detect_image(image) #检测
        result = np.asarray(image)
        # 计算fps
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        # 画出fps
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 255, 255), thickness=2) # 255改成白色了
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()
