import dlib
import numpy as np
import cv2
import os
import pandas as pd
import csv
import time

from Mast import inf_Mast
from huoti import Get_ER
from os_handle import *

# Dlib 正向人脸检测器 / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()
# Dlib 人脸 landmark 特征点检测器 / Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
# Dlib Resnet 人脸识别模型，提取 128D 的特征矢量 / Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


class Timer():
    def __init__(self):
        self.start_time = time.time()
        self.result = False

    def run(self):
        self.end_time = time.time()
        T = int(self.end_time - self.start_time)
        if T % 5 == 4:
            self.result = True
        else:
            self.result = False
        return self.result


class TRT:
    def __init__(self):
        self.font = cv2.FONT_ITALIC
        self.mood = ''
        self.IsMask = False
        self.face = None
        # cnt for frame
        self.frame_cnt = 0
        # 配置活体检测参数
        self.IsGone_flag = False
        self.IsGoing_flag = False
        self.NotIsPhoto = True
        self.start_time = 0.0
        self.end_time = 0.0
        self.Counter = 0
        self.Total = 0
        # 用来存储所有录入人脸特征的数组 / Save the features of faces in the database
        self.features_known_list = []
        # 用来存储录入人脸名字 / Save the name of faces in the database
        self.name_known_list = []

        # 用来存储上一帧和当前帧检测出目标的名字 / List to save names of objects in frame N-1 and N
        self.last1_frame_name = None
        self.last2_frame_name = None
        self.last3_frame_name = None
        self.current_frame_face_name = None

        # 上一帧和当前帧中人脸数的计数器 / cnt for faces in frame N-1 and N
        self.last_frame_faces_cnt = 0
        self.current_frame_face_cnt = 0

        # 用来存放进行识别时候对比的欧氏距离 / Save the e-distance for faceX when recognizing
        self.current_frame_face_X_e_distance_list = []

        # 存储当前摄像头中捕获到的所有人脸的坐标名字 / Save the positions and names of current faces captured
        self.current_frame_face_position = None
        # 存储当前摄像头中捕获到的人脸特征 / Save the features of people in current frame
        self.current_frame_face_features_list = []
        #按键操作标识符
        self.press_n_flag = False
        self.press_g_flag = True
        #地址路径
        self.photo_path = "./data/data_faces_from_camera/"
    # 从 "features_all.csv" 读取录入人脸特征 / Get known faces from "features_all.csv"
    def rebuild_database_AND_name(self):
        del_list = []
        if os.path.isfile("data/features_all.csv"):
            os.remove("data/features_all.csv")
        if os.path.isfile("data/name_all.csv"):
            os.remove("data/name_all.csv")

        self.name_known_list = os.listdir(self.photo_path)

        for l_name in self.name_known_list:
            Is_ok, features_exist_name = return_features_mean_personX(self.photo_path+l_name)
            if Is_ok:
                self.features_known_list.append(features_exist_name)
            else:
                del_list.append(l_name)

        if len(del_list)>0:
            for i in del_list:
                self.name_known_list.pop(self.name_known_list.index(i))

        print(len(self.name_known_list),self.name_known_list)
        print(len(self.features_known_list))



    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            # 将逗号分隔值（csv）文件读入DataFrame
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                for j in range(0, 128):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.features_known_list.append(features_someone_arr)
                # self.name_known_list.append("Person_" + str(i + 1))

            print(self.name_known_list)
            print("Faces in Database：", len(self.features_known_list))

            if len(self.name_known_list) == len(self.features_known_list):
                return True
            else:
                print("人脸名字与特征向量不匹配，开始重构人脸数据")
                self.features_known_list=[]
                self.rebuild_database_AND_name()
                return True
        else:
            print("不存在数据文件，开始重构人脸数据")
            self.rebuild_database_AND_name()
            return True


    # 计算两个128D向量间的欧式距离 / Compute the e-distance between two 128D features
    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    def learning_face(self, shape):
        # 分析点的位置关系来作为表情识别的依据
        mouth_width = abs(shape.part(54).x - shape.part(48).x) / self.face_width  # 嘴巴咧开程度
        mouth_higth = abs(shape.part(66).y - shape.part(62).y) / self.face_width  # 嘴巴张开程度
        # 眼睛睁开程度
        eye_sum = (shape.part(41).y - shape.part(37).y + shape.part(40).y - shape.part(38).y + shape.part(
            47).y - shape.part(43).y + shape.part(46).y - shape.part(44).y)
        eye_hight = (eye_sum / 4) / self.face_higth
        # 分情况讨论
        # 张嘴，可能是开心或者惊讶
        if mouth_width >= 0.35 or mouth_higth >= 0.023:
            self.mood = 'Happy'
        if mouth_width <= 0.35 and mouth_higth <= 0.023:
            self.mood = 'Netural'
        if mouth_higth >= 0.031 and eye_hight > 0.040:
            if mouth_width <= 0.35:
                self.mood = 'Surprise'

        print("mouth_higth:{}  mouth_width:{}  eye_hight:{}".format(round(mouth_higth, 3), round(mouth_width, 3), round(eye_hight, 3)))
        print(self.mood)

        # 生成的 cv2 window 上面添加说明文字 / putText on cv2 window

    def draw_note(self, img_rd):
        # 添加说明 / Add some statements
        # cv2.putText(img_rd, "Face Recognizer with OT (one person)", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        if self.current_frame_face_name == self.last1_frame_name \
                and self.last2_frame_name == self.last1_frame_name \
                and self.current_frame_face_name != None:
            if self.NotIsPhoto and not self.IsMask:
                cv2.putText(img_rd, self.current_frame_face_name + " " + self.mood, (20, 20), self.font, 0.8,
                            (0, 0, 255), 1, cv2.LINE_AA)
            elif self.NotIsPhoto and self.IsMask:
                cv2.putText(img_rd, self.current_frame_face_name, (20, 20), self.font, 0.8,
                            (0, 0, 255), 1, cv2.LINE_AA)
            elif not self.NotIsPhoto:
                cv2.putText(img_rd, "no face", (20, 20), self.font, 0.8,
                            (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    def Get_mostsimilar_person_num(self, img_rd, Boundary=0.44):
        self.current_frame_face_X_e_distance_list = []
        shape = predictor(img_rd, self.face)
        self.current_frame_face_features = face_reco_model.compute_face_descriptor(img_rd, shape)
        # 2.2.2.3 对于某张人脸，遍历所有存储的人脸特征
        # For every faces detected, compare the faces in the database
        for i in range(len(self.features_known_list)):
            # 如果 person_X 数据不为空
            if str(self.features_known_list[i][0]) != '0.0':
                print("            >>> with person", str(i + 1), "the e distance: ", end='')
                e_distance_tmp = self.return_euclidean_distance(
                    self.current_frame_face_features,
                    self.features_known_list[i])
                print(e_distance_tmp)
                self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
            else:
                # 空数据 person_X
                self.current_frame_face_X_e_distance_list.append(999999999)
        # 2.2.2.4 寻找出最小的欧式距离匹配 / Find the one with minimum e distance
        similar_person_num = self.current_frame_face_X_e_distance_list.index(
            min(self.current_frame_face_X_e_distance_list))
        if min(self.current_frame_face_X_e_distance_list) < Boundary:
            print("            >>> recognition result for face " + str(1) + ": " +
                  self.name_known_list[similar_person_num])
            return self.name_known_list[similar_person_num]
        else:
            print("            >>> recognition result for face " + str(1) + ": " + "unknown")
            return 'unknown'

    # 处理获取的视频流，进行人脸识别 / Face detection and recognition wit OT from input video stream
    def process(self, stream):
        timer = Timer()
        # 1. 读取存放所有人脸特征的 csv / Get faces known from "features.all.csv"
        if self.get_face_database():
            while stream.isOpened():
                flag, img_rd = stream.read()
                self.frame_cnt += 1
                # 跳帧处理 减少卡顿
                if self.frame_cnt % 4 != 0 and not self.IsGoing_flag:
                    continue
                kk = cv2.waitKey(1)
                print(">>> Frame " + str(self.frame_cnt) + " starts")

                # 更新当前帧的数据
                self.last_frame_faces_cnt = self.current_frame_face_cnt
                self.current_frame_face_cnt = 0
                self.last3_frame_name = self.last2_frame_name
                self.last2_frame_name = self.last1_frame_name
                self.last1_frame_name = self.current_frame_face_name

                # 判断是否戴口罩
                info = inf_Mast(cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB))
                img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
                #print(info)
                if info != None:
                    self.current_frame_face_cnt = info[5]
                    cv2.rectangle(img_rd,
                                  tuple([info[1], info[2]]),
                                  tuple([info[3], info[4]]),
                                  (255, 255, 255), 2)
                    if info[0] == 0:
                        cv2.putText(img_rd, "Mask", (20, 200), self.font, 0.8,
                                    (0, 0, 255), 1, cv2.LINE_AA)
                        self.IsMask = True
                        self.mood = ''
                        self.face = dlib.rectangle(info[1], info[2], info[3], info[4])
                    elif info[0] == 1:
                        faces = detector(img_gray, 0)
                        #print(len(faces))
                        for k, d in enumerate(faces):
                            self.face = d
                        self.IsMask = False
                        if self.face == None:
                            cv2.imshow("camera", img_rd)
                            continue
                        cv2.putText(img_rd, "No Mask", (20, 200), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
                else:
                    self.face = None
                    self.current_frame_face_cnt = 0
                    print("   >>> scene 2.1 人脸消失, 当前帧中没有人脸 / no guy in this frame!!!")
                    # clear list of names and
                    self.current_frame_face_cnt = 0
                    self.current_frame_face_name = None
                    self.current_frame_face_features_list = []
                    self.IsGone_flag = False
                    self.NotIsPhoto = True
                    self.mood = ''

                print("   >>> current_frame_face_cnt: ", self.current_frame_face_cnt)
                if not self.IsGone_flag \
                        and not self.IsMask \
                        and not self.IsGoing_flag \
                        and self.last3_frame_name == self.last1_frame_name \
                        and self.current_frame_face_cnt == self.last_frame_faces_cnt \
                        and self.current_frame_face_name in ['1Xiaoyu', '2Xiaohui']:
                    self.start_time = time.time()
                    self.IsGoing_flag = True

                if self.IsGoing_flag:
                    Info = Get_ER(cv2.cvtColor(img_rd, cv2.COLOR_BGR2GRAY))
                    if Info != None:
                        cv2.drawContours(img_rd, [cv2.convexHull(Info[1])], -1, (0, 255, 0), 1)
                        cv2.drawContours(img_rd, [cv2.convexHull(Info[2])], -1, (0, 255, 0), 1)
                        if Info[0] < 0.2:  # 眼睛长宽比：0.2
                            self.Counter += 1
                        else:
                            # 如果连续3次都小于阈值，则表示进行了一次眨眼活动
                            if self.Counter >= 2:  # 阈值：2
                                self.Total += 1
                            # 重置眼帧计数器
                            self.Counter = 0

                if self.Total >= 2:
                    self.start_time = 0.0
                    self.IsGone_flag = True
                    self.IsGoing_flag = False
                    self.NotIsPhoto = True
                    self.Counter = 0
                    self.Total = 0
                    self.end_time = time.time()

                if time.time() - self.start_time > 8 \
                        and self.IsGoing_flag:
                    self.start_time = 0.0
                    self.IsGone_flag = True
                    self.IsGoing_flag = False
                    self.NotIsPhoto = False
                    self.Counter = 0
                    self.Total = 0
                    self.end_time = time.time()

                if self.IsGoing_flag:
                    cv2.putText(img_rd, "Testing living detection...", (300, 50), self.font, 0.8, (0, 0, 255), 1,
                                cv2.LINE_AA)
                    cv2.putText(img_rd, "Please blink your eyes slowly(Three times)!", (300, 75), self.font, 0.8,
                                (0, 0, 255), 1,
                                cv2.LINE_AA)
                else:
                    cv2.putText(img_rd, "end livig detection !", (300, 50), self.font, 0.8, (0, 255, 0), 1,
                                cv2.LINE_AA)
                # 2.1 If cnt not changes, 1->1 or 0->0
                if self.current_frame_face_cnt == self.last_frame_faces_cnt \
                        and self.last3_frame_name == self.last1_frame_name \
                        and self.current_frame_face_name != None:

                    print("   >>> scene 1: 当前帧和上一帧相比没有发生人脸数变化 / no faces cnt changes in this frame!!!")
                    print("   >>> self.current_frame_face_name:        ", self.current_frame_face_name)
                    # One face in this frame
                    if len(faces) == 1:
                        if self.NotIsPhoto and not self.IsMask:
                            for k, d in enumerate(faces):
                                self.face_higth = (d.bottom() - d.top())
                                self.face_width = (d.right() - d.left())
                                shape = predictor(img_gray, d)
                                self.learning_face(shape)
                # 2.2 if cnt of faces changes, 0->1 or 1->0
                else:
                    print("   >>> scene 2: 当前帧和上一帧相比人脸数发生变化 / Faces cnt changes in this frame")
                    self.current_frame_face_position_list = []
                    self.current_frame_face_X_e_distance_list = []
                    # 2.2.2 face cnt: 0->1, get the new face
                    if self.current_frame_face_cnt == 1:
                        print("   >>> scene 2.2 出现人脸，进行人脸识别 / Get person in this frame and do face recognition")
                        if self.IsMask:
                            self.current_frame_face_name = self.Get_mostsimilar_person_num(img_rd, 2)
                        else:
                            self.current_frame_face_name = self.Get_mostsimilar_person_num(img_rd)
                        print(self.current_frame_face_name)

                if timer.run():
                    if self.current_frame_face_cnt == 1:
                        print("   >>> scene 3.1 刷新人脸 / Get person in this frame and do face recognition again")
                        if self.IsMask:
                            self.current_frame_face_name = self.Get_mostsimilar_person_num(img_rd, 2)
                        else:
                            self.current_frame_face_name = self.Get_mostsimilar_person_num(img_rd)

                # 3. 生成的窗口添加说明文字 / Add note on cv2 window
                self.draw_note(img_rd)

                #添加人脸信息
                if kk == ord('n'):
                    if not self.press_g_flag:
                        print("请按“g” ---- 更新新建的人脸")
                    else:
                        self.press_g_flag = False
                        name = input("新建的人脸文件夹 / Create folders: ")
                        current_face_dir = self.photo_path + name
                        mkdir(current_face_dir)
                        self.name_known_list.append(name)
                        ss_cnt = 0
                        self.press_n_flag = True
                        # 按下 's' 保存截图
                        if kk == ord('s'):
                            if self.press_n_flag:
                                if self.face == None:
                                    ss_cnt += 1
                                    cv2.imwrite(current_face_dir + "/" + name + str(ss_cnt) + ".jpg", img_rd)
                                    print("已保存 {}".format(current_face_dir + "/" + name + str(ss_cnt) + ".jpg"))
                                else:
                                    print("图片不符合要求")
                            else:
                                print("当前没有添加人脸信息请求！ 请先按“n” ")

                if kk == ord('g'):
                    if self.press_n_flag:
                        Is_ok, features_new_name = return_features_mean_personX(current_face_dir)
                        if Is_ok:
                            self.features_known_list.append(features_new_name)
                        else:
                            self.name_known_list.pop()
                    else:
                        print("当前暂无新建人脸信息！！！  如有需要请先按“n”")

                if kk == ord('q'):
                    # 保存人名信息
                    with open("./data/name_all.csv", "w", newline="") as csvfile1:
                        writer = csv.writer(csvfile1)
                        writer.writerow(self.name_known_list)
                    # 保存特征向量的值
                    with open("./data/features_all.csv", "w", newline="") as csvfile2:
                        writer = csv.writer(csvfile2)
                        for feature in self.features_known_list:
                            writer.writerow(feature)
                    break
                cv2.namedWindow("camera", 1)
                cv2.imshow("camera", img_rd)

                print(">>> Frame ends\n\n")



    def run(self):
        self.name_known_list= read_name_list()
        cap = cv2.VideoCapture(0)
        self.process(cap)
        cap.release()
        cv2.destroyAllWindows()


def main():
    TRT001 = TRT()
    TRT001.run()


if __name__ == '__main__':
    main()
