from scipy.spatial import distance as dist
import dlib
import numpy as np
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords

def eye_aspect_ratio(eye):
	# 垂直眼标志（X，Y）坐标
	A = dist.euclidean(eye[1], eye[5])  # 计算两个集合之间的欧式距离
	B = dist.euclidean(eye[2], eye[4])
	# 计算水平之间的欧几里得距离
	# 水平眼标志（X，Y）坐标
	C = dist.euclidean(eye[0], eye[3])
	# 眼睛长宽比的计算
	ear = (A + B) / (2.0 * C)
	# 返回眼睛的长宽比
	return ear

def Get_ER(gray):
	output_info = None
	rects = detector(gray, 0)
	for rect in rects:
		shape = predictor(gray, rect)
		shape = shape_to_np(shape)
		leftEye = shape[42:48]
		rightEye = shape[36:42]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ER = (leftEAR + rightEAR) / 2.0

		output_info = [ER,leftEye,rightEye]
	return output_info