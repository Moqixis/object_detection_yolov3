import re
import os

# dir_project = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # 获取上上级目录
dir_project = 'mycode/test2'
dir_result = '/result/result.txt'  # yolo批量处理结果的目录
dir_detection_results = '/mAP/input/detection-results'  # detection-results目录
surplus = 'mycode/test2/VOCdevkit/VOC2007/JPEGImages/'  # result.txt文件中图片名称多余的部分

if __name__ == '__main__':
    with open(dir_project + dir_result, 'r') as f:  # 打开文件
        filename = f.readlines()  # 读取文件

    for i in range(len(filename)):
        filename[i] = re.sub(surplus, '', filename[i])        # 去除文件名多余的部分

    for i in range(len(filename)):  # 中按行存放的检测内容，为列表的形式
        r = filename[i].split('.jpg ')
        file = open(dir_project + dir_detection_results + '/' + r[0] + '.txt', 'w')
        t = r[1].split(';')
        # 去除空格和换行
        t.remove('\n')

        if len(t) == 0:            # 如果没有对象
            file.write('')
        else:
            for k in range(len(t)):
                file.write(t[k] + '\n')
