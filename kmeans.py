import numpy as np


class YOLO_Kmeans:

    def __init__(self, cluster_number, filename): # cluster_number聚类个数
        self.cluster_number = cluster_number
        self.filename = "2007_train.txt"

    def iou(self, boxes, clusters):  # 1 box -> k clusters 计算IoU，返回值result是[boxes.size(),k]
        n = boxes.shape[0]
        k = self.cluster_number
        # 把box_area整理成n行k列的形式
        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n]) # data为要扩展的数据, x扩展行数, y扩展列数
        cluster_area = np.reshape(cluster_area, (n, k))
        # 把box和cluster的宽都整理成n行k列的形式，并把两者做比较，最后还是一个n行k列的形式，这个过程其实在比较box和两个cluster的宽，并选出小的
        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix) # 交集

        result = inter_area / (box_area + cluster_area - inter_area) # 交并比IoU
        return result

    def avg_iou(self, boxes, clusters): # 取每个box与其计算有最大的iou的值，计算所有最大的iou的值的平均值
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)]) 
        return accuracy

    def kmeans(self, boxes, k, dist=np.median): # dist为传入kmeans函数的函数，np.median函数是计算沿指定轴的中位数
        box_number = boxes.shape[0]             # 返回boxes数组的纵向维度，即数据中总共有多少个框
        distances = np.empty((box_number, k))   # 创建一个没有任何具体值的数组
        last_nearest = np.zeros((box_number,))  # 创建一个全0的数组
        np.random.seed()
        ''' 
        np.random.choice(a,size=None,replace=true,p=None)  
        从a中随机选取size个元素组成数组，replace为true表示可以重复选择，为false表示不可以重复选元素，
        p为数组，与a相对应，表示取a中每个元素的概率大小，默认选取每个元素的概率相同
        '''
        clusters = boxes[np.random.choice(  # 初始化簇中心，随机选取k个宽高作为簇中心
            box_number, k, replace=False)]  # init k clusters
        while True:

            distances = 1 - self.iou(boxes, clusters) # 距离为 1 - IoU

            current_nearest = np.argmin(distances, axis=1) # 取iou值最小的的点，横轴为axis=1
            if (last_nearest == current_nearest).all(): # 比较本次与上一次的k个簇中心是否变化
                break  # clusters won't change 没有变化则停止更新
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0) # 取分到cluster组的元素的中值(dist函数即np.median是取中位数)

            last_nearest = current_nearest

        return clusters

    def result2txt(self, data):  # 将计算结果写进yolo_anchors.txt
        f = open("yolo_anchors.txt", 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2boxes(self):   # 从训练集文件中读取并计算实际的宽 高
        f = open(self.filename, 'r')
        dataSet = []
        for line in f:
            infos = line.split(" ")
            length = len(infos)
            for i in range(1, length): # in fors中的数据格式[1,2,100,110] ,数据框左上角坐标x，y，数据框右下角坐标x，y
                width = int(infos[i].split(",")[2]) - \
                    int(infos[i].split(",")[0])
                height = int(infos[i].split(",")[3]) - \
                    int(infos[i].split(",")[1])
                dataSet.append([width, height])
        result = np.array(dataSet)     # 利用dataset列表创建一个二维数组result
        f.close()
        return result

    def txt2clusters(self):             # 计算 anchors
        all_boxes = self.txt2boxes()    # 加载实际数据 宽高
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])] # 对9个中心值进行排序
        self.result2txt(result)
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(
            self.avg_iou(all_boxes, result) * 100))


if __name__ == "__main__":
    cluster_number = 6
    filename = "2007_train.txt"         # 训练集文件，格式为： 路径 [左上 右下 类别](5个值) ...
    kmeans = YOLO_Kmeans(cluster_number, filename)
    kmeans.txt2clusters()
