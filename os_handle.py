import dlib
import numpy as np
import os
from skimage import io
import csv

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor('./data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Dlib Resnet 人脸识别模型，提取 128D 的特征矢量 / Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("./data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

def mkdir(path):
    # 新建文件夹 / Create folders to save faces images and csv
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)

def read_name_list():
    name_list = []
    if os.path.isfile("data/name_all.csv"):
       with open("./data/name_all.csv") as csvfile:
           reader = csv.reader(csvfile)
           for name in reader:
               if name == None:
                   name_list = []
               else:
                   name_list = name
    return name_list



def return_128d_features(path_img):
    img_rd = io.imread(path_img)
    faces = detector(img_rd, 1)
    print("%-40s %-20s" % ("检测到人脸的图像 / Image with faces detected:", path_img), '\n')

    # 因为有可能截下来的人脸再去检测，检测不出来人脸了, 所以要确保是 检测到人脸的人脸图像拿去算特征
    # For photos of faces saved, we need to make sure that we can detect faces from the cropped images
    if len(faces) != 0:
        shape = predictor(img_rd, faces[0])
        face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
    else:
        face_descriptor = 0
    return face_descriptor

def return_features_mean_personX(path_faces_personX):
    features_list_personX = []
    photos_list = os.listdir(path_faces_personX)
    if photos_list:
        for photo in photos_list:
            # 调用 return_128d_features() 得到 128D 特征 / Get 128D features for single image of personX
            print("%-40s %-20s" % ("正在读的人脸图像 / Reading image:", path_faces_personX + "/" + photo))
            features_128d = return_128d_features(path_faces_personX + "/" + photo)
            # 遇到没有检测出人脸的图片跳过 / Jump if no face detected from image
            if features_128d == 0:
                continue
            else:
                features_list_personX.append(features_128d)
    else:
        os.rmdir(path_faces_personX)
        print("文件夹内图像文件为空 / Warning: No images in " + path_faces_personX + '/')
        print("已删除该目录")
        return False,[0]

    # 计算 128D 特征的均值 / Compute the mean
    # personX 的 N 张图像 x 128D -> 1 x 128D
    if features_list_personX:
        features_mean_personX = np.array(features_list_personX).mean(axis=0).tolist()
    else:
        io.rmtree(path_faces_personX)
        for photo in photos_list:
            os.remove(path_faces_personX + "/" + photo)
        os.rmdir(path_faces_personX)
        return False,[0]
    return True,features_mean_personX