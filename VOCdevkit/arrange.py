import os
import os.path as osp
import shutil
from PIL import Image, ImageEnhance
import cv2

# __file__是当前文件
BASEDIR = osp.dirname(osp.abspath(__file__))

IMGDIR = osp.join(BASEDIR, 'image')
LABDIR = osp.join(BASEDIR, 'image')
SOUDIR = osp.join(BASEDIR, 'images')
OUTDIR = osp.join(BASEDIR, 'labels')
# 如果存在会删掉。。从头运行的时候记得取消注释!!!
if osp.exists(OUTDIR):
    shutil.rmtree(OUTDIR)
os.makedirs(OUTDIR)
if osp.exists(SOUDIR):
    shutil.rmtree(SOUDIR)
os.makedirs(SOUDIR)
    
# 讲json文件保存为png
def arrange():
    i = 0
    imgdirs = [name for name in os.listdir(IMGDIR)]
    for imgdir in imgdirs:
        imgnames = [name for name in os.listdir(IMGDIR+'\\'+imgdir) if name.split('.')[-1] in ['png', 'jpg', 'tif', 'bmp']]  # 按扩展名过滤
        for imgname in imgnames:   
            path_json = osp.join(LABDIR+'\\'+imgdir, imgname.split('.')[0]+'.json')  # 找到对应名字的json标注文件
            new_path_json = OUTDIR + '\\image' + str(i) + '.json'
            shutil.copyfile(path_json, new_path_json)
            JPG_file = IMGDIR + '\\' + imgdir + "\\" + imgname
            new_JPG_file = SOUDIR + '\\image' + str(i) + ".jpg"
            shutil.copyfile(JPG_file, new_JPG_file)
            i += 1
            # break
    print(i)

if __name__ == '__main__':
    arrange()