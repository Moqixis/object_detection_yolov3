import re
import os

# dir_project = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # 获取上上级目录
dir_project = 'mycode/test2'
dir_ground_truth = '/mAP/input/ground-truth'  # ground-truth目录
surplus = '/home/lihuiqian/mycode/test2/VOCdevkit/VOC2007/JPEGImages/'  # result.txt文件中图片名称多余的部分

if __name__ == '__main__':
    with open(dir_project + '/2007_test.txt', 'r') as f:  # 打开文件
        filename = f.readlines()  # 读取文件
        # print(filename)

    for i in range(len(filename)):
        filename[i] = re.sub(surplus, '', filename[i])    # 去除文件名多余的部分

    for i in range(len(filename)):  # 中按行存放的检测内容，为列表的形式
        r = filename[i].split('.jpg ')
        print(r[0])
        file = open(dir_project + dir_ground_truth + '/' + r[0] + '.txt', 'w')
        t = r[1].split(' ')

        for j in range(len(t)):
            class_t = t[j].split(',')[-1]
            pos_t = t[j].split(',')
            if class_t == '0' or class_t == '0\n':
                file.write('polyp ' + pos_t[0] + ' ' + pos_t[1] + ' '+ pos_t[2] + ' '+ pos_t[3] + '\n')
