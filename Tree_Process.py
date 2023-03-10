import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


class TreeProcess(object):
    def __init__(self, path):

        self.Canny_Img = None
        self.OTSU_ret = None
        self.path = path
        self.size = (640, 480)
        self.original_img = cv2.resize(cv2.imread(self.path), self.size)
        self.original_img_copy = self.original_img.copy()

        self.Lab_Img = None
        self.channelA_Img = None

        self.first_close_img = None  # 图像第一次闭运算的结果
        self.Img_PreProcess()  # 对图像进行预处理

        self.MinYRectangle_Img = None
        self.rectangle_h = None
        self.rectangle_w = None
        self.componentMask_Img = None

    def set_Img_Path(self, _path):
        self.path = _path
        self.set_Img_size(self.size)

    def set_Img_size(self, _size):
        """
        重新设置原图像尺寸并初始化
        :param _size:
        """
        self.size = _size
        self.original_img = cv2.resize(self.original_img, _size)
        self.Img_PreProcess()  # 对图像进行预处理

    def get_Lab_Img(self):
        """
        获得转至Lab通道的图像
        :return: Lab通道图像
        """
        self.Lab_Img = cv2.cvtColor(self.original_img, cv2.COLOR_RGB2Lab)
        return self.Lab_Img

    def get_channelA_Img(self):
        """
        获得A通道图像
        :return: A通道图像
        """
        if self.Lab_Img is None:
            self.get_Lab_Img()
        L, a, b = cv2.split(self.Lab_Img)
        self.channelA_Img = a
        return self.channelA_Img

    def Img_PreProcess(self):
        """
        图像预处理，将图像经尺寸转换、Lab转换、通道分离、对a通道进行高斯模糊后，进行自适应直方图均匀化，进行Otsu算法的阈值转换后再进行闭运算得到的图像
        :param
        :return: close : 闭运算图像
        """
        a = self.get_channelA_Img()
        # 进行高斯模糊去除部分噪音
        a_Gaussian = cv2.GaussianBlur(a, (3, 3), 1)
        # 第一次OTSU算法得到阈值，可利用该阈值将样木与部分背景分离
        Gaussian_ret, Gaussian_threshold = cv2.threshold(a_Gaussian, 0, 255,
                                                         cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        clahe = cv2.createCLAHE(Gaussian_ret / 26, (8, 8))
        """
        参数1：灰度值出现的次数，若某灰度值出现的次数超过该参数，则将相减部分均匀的分配给其他像素
        参数2：图像会被划分的size
        https://www.cnblogs.com/silence-cho/p/11006958.html
        """
        a_dst = clahe.apply(a_Gaussian)
        cv2.imshow('a_dst', a_dst)
        cv2.waitKey()
        # 对比度增强后，再次利用OTSU算法进行阈值转换
        self.OTSU_ret, a_dst_Gaussian_threshold = cv2.threshold(a_dst, 0, 250,
                                                                cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        # 设置卷积核的shape
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # 利用卷积核进行闭运算去除掉部分不需要的信息
        self.first_close_img = cv2.morphologyEx(a_dst_Gaussian_threshold, cv2.MORPH_OPEN, kernel,
                                                          iterations=3)
        cv2.imshow('close', self.first_close_img)
        cv2.waitKey(0)
        return self.first_close_img

    def get_first_close_img(self):
        return self.first_close_img

    def get_MinYRectangle_Img(self):
        """
        得到Y坐标最小的的框选矩形（以左下角为原点）
        :return: Y最小（最近的树木）的闭运算图像
                 w: 所框选矩形的宽
                 h: 所框选矩形的高
        """
        # 输入闭运算后的图像
        contours, hierarchy = cv2.findContours(self.first_close_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 搜索距离y轴最低的树木
        num = len(contours)
        rect_list = []
        for i in range(num):
            area = cv2.contourArea(contours[i], oriented=False)
            if self.size[0] * self.size[1] / 100 < area < self.size[0] * self.size[1] / 3:  # 限定轮廓的面积
                # print('筛选面积最大值', self.size[0] * self.size[1] / 3)
                # print(area)
                rect_list.append(cv2.boundingRect(contours[i]))  # 求出最小外包直角矩形并记录

        if len(rect_list) > 0:
            # 计算y坐标并记录
            y_list = [rect[1] + rect[3] for rect in rect_list]
            # 得到最大y坐标的索引,最大y坐标代表最靠近相机
            index = y_list.index(max(y_list))
            x = rect_list[index][0]
            y = rect_list[index][1]
            self.rectangle_w = rect_list[index][2]
            self.rectangle_h = rect_list[index][3]
            cv2.rectangle(self.original_img_copy, (x, y), (x + self.rectangle_w, y + self.rectangle_h),
                          (0, 255, 0))  # 在复制图上画框

            # print(len(y_list))
            mask_1 = np.ones(self.original_img.shape[0:2], dtype="uint8")
            mask_1[y:y + self.rectangle_h, x:x + self.rectangle_w] = 255  # 将框选区域的像素值设为白色

            self.MinYRectangle_Img = cv2.bitwise_and(self.first_close_img, mask_1)  # 进行与运算，去除背景

        return self.MinYRectangle_Img, self.rectangle_w, self.rectangle_h

    def get_Max_connectedComponents(self):
        """
        得到最大连通区域（在这一版本中会出现核心已转储的段错误，cv库的问题）
        :return:
        """

        if self.MinYRectangle_Img is not None:
            output = cv2.connectedComponentsWithStats(self.MinYRectangle_Img, connectivity=4, ltype=cv2.CV_32S)
        else:
            output = cv2.connectedComponentsWithStats(self.first_close_img, connectivity=4, ltype=cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        componentMask_list = []
        for i in range(1, numLabels):  # 忽略背景
            x = stats[i, cv2.CC_STAT_LEFT]  # [i, 0]
            y = stats[i, cv2.CC_STAT_TOP]  # [i, 1]
            componentMask_w = stats[i, cv2.CC_STAT_WIDTH]  # [i, 2]
            componentMask_h = stats[i, cv2.CC_STAT_HEIGHT]  # [i, 3]
            area = stats[i, cv2.CC_STAT_AREA]  # [i, 4]
            if self.rectangle_w and self.rectangle_h is not None:
                # 确保宽高以及面积既不太大也不太小
                keepWidth = self.rectangle_w / 3 < componentMask_w
                keepHeight = self.rectangle_h / 3 < componentMask_h
                keepArea = self.rectangle_w * self.rectangle_h / 9 < area
            else:
                keepWidth = self.size[1] / 3 < componentMask_w
                keepHeight = self.size[0] / 3 < componentMask_h
                keepArea = self.size[1] * self.size[0] / 9 < area

            if all((keepWidth, keepHeight, keepArea)):
                print(componentMask_w, componentMask_h, area)
                print("[INFO] keep connected component '{}'".format(i))
                componentMask_list.append((labels == i).astype("uint8") * 255)
            # 我使用print语句显示每个连接组件的宽度、高度和面积，
            # 得到符合要求的连通区域

        componentMaskImg = self.channelA_Img.copy()

        for componentMask in componentMask_list:
            # 对连通区域进行一次阈值转换，将白的变为黑的，用于去除原图像大部分不需要的背景
            ret, componentMask_threshold = cv2.threshold(componentMask, 127, 255, cv2.THRESH_BINARY)
            # 掩膜操作，去除背景
            componentMaskImg = cv2.bitwise_and(componentMaskImg, componentMask_threshold)
        self.componentMask_Img = componentMaskImg
        return self.componentMask_Img

    def get_componentMask_Img_Canny(self):
        self.Canny_Img = cv2.Canny(self.componentMask_Img, threshold1=self.OTSU_ret / 2, threshold2=80)
        return self.Canny_Img

    def get_MinY_Line_Gap(self, _minLineLength, _maxLineGap):
        """
        在canny边缘检测后的图像中画出最靠近底部的两条线
        :param _minLineLength: 线的最短长度，比这个线短的都会被忽略
        :param _maxLineGap: 两条线之间的最大间隔，如果小于此值，这两条线就会被看成一条线
        :return:Gap: 两条直线间的像素距离
        """
        Gap = 0.
        # 利用霍夫直线检测检测边缘图像符合条件的直线
        lines = cv2.HoughLinesP(self.Canny_Img, 1, np.pi / 180, 100, minLineLength=_minLineLength,
                                maxLineGap=_maxLineGap)
        """
        参数2rho:对应直线搜索的步长
        参数3theta：步长为π/180的角来搜索所有可能的直线。
        参数4threshold：是经过某一点曲线的数量的阈值，超过这个阈值，就表示这个交点所代表的参数对(rho, theta)在原图像中为一条直线
        """
        y_dic = {}
        for line in lines:
            x1, y1, x2, y2 = line[0]
            theta = 0.
            try:
                theta = (math.atan((y2 - y1) / (x2 - x1))) * 57.3
            except Exception as e:
                print("错误信息:", e)
            # print(theta)
            if abs(theta) > 85:  # 筛选出角度大于85度的
                # print(line[0])
                y_dic.update({y2: (x1, y1, x2, y2)})
                y_dic.update({y1: (x1, y1, x2, y2)})
                cv2.line(self.original_img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # y_list = sorted(y_dic)
        # print("y2的坐标列表为：", y_dic)

        # if len(y_list) >= 2:
        #     x11 = y_dic.get(y_list[len(y_list) - 2])[0]
        #     y11 = y_dic.get(y_list[len(y_list) - 2])[1]
        #     x12 = y_dic.get(y_list[len(y_list) - 2])[2]
        #     y12 = y_dic.get(y_list[len(y_list) - 2])[3]
        #     x21 = y_dic.get(y_list[len(y_list) - 1])[0]
        #     y21 = y_dic.get(y_list[len(y_list) - 1])[1]
        #     x22 = y_dic.get(y_list[len(y_list) - 1])[2]
        #     y22 = y_dic.get(y_list[len(y_list) - 1])[3]
        #     print("第一条直线为", x11, y11, x12, y12)
        #     print("第二条直线为", x21, y21, x22, y22)
        #
        #     cv2.line(self.original_img_copy, (x11, y11), (x12, y12), (0, 0, 255), 2)
        #     cv2.line(self.original_img_copy, (x21, y21), (x22, y22), (0, 0, 255), 2)
        #
        #     Gap = abs(((x11 + x12) / 2) - ((x21 + x22) / 2))
        return Gap


if __name__ == "__main__":
    tree = TreeProcess('img/tree3.jpeg')

    # 得到Y坐标最小的的框选矩形（以左下角为原点）
    MinYRectangle_Img, Rectangle_Img_w, Rectangle_Img_h = tree.get_MinYRectangle_Img()
    cv2.imshow("MinYRectangle_Img", MinYRectangle_Img)
    cv2.waitKey(0)
    # componentMask_Img = tree.get_Max_connectedComponents()
    # cv2.imshow("componentMask_Img", componentMask_Img)
    # cv2.waitKey(0)
    # Canny = tree.get_componentMask_Img_Canny()
    # cv2.imshow("Canny", Canny)
    # cv2.waitKey(0)
    # minLineLength = tree.size[0] / 20
    # maxLineGap = tree.size[0] / 5
    # gap = tree.get_MinY_Line_Gap(minLineLength, maxLineGap)
    # cv2.imshow("Characters", tree.original_img_copy)
    # cv2.waitKey(0)
