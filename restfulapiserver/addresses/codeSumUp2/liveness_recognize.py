# USAGE
#  python liveness_recognize.py --model liveness.model --le le.pickle --detector face_detector --embedding_model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le2 output/le2.pickle

# import the necessary packages
from imutils.video import VideoStream
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
from imutils.video import FPS
people = {}

def put(people, name, value):
	if name not in people:
		people[name] = value
	else:
		people[name] += value
#  tf 버전 호환문제
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

# 파라메터 구문 분석
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True, help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=True, help="path to label encoder")
ap.add_argument("-d", "--detector", type=str, required=True, help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-e", "--embedding_model", required=True, help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True, help="path to model trained to recognize faces")
ap.add_argument("-l2", "--le2", type=str, required=True, help="path to label encoder")


args = vars(ap.parse_args())

# 얼굴 탐지기 로딩
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# 진짜 얼굴 탐지 모델 및 레이블 로딩
print("[INFO] loading liveness detector...")
model = load_model(args["model"])
le = pickle.loads(open(args["le"], "rb").read())

# 비디오 스트림을 초기화
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# 얼굴 인식기 로딩 (128-D 얼굴 인식을 계산하기 위해 사전 훈련된 Torch DL 모델)
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# 레이블 인코더와 함께 실제 얼굴 인식 모델 로딩 (Linear SVM얼굴 인식모델)
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le2 = pickle.loads(open(args["le2"], "rb").read())
# 비디오 스트림에서 프레임 반복
while True:
	# 비디오 스트림 프레임의 이미지 크기를 600 픽셀 너비로 조정
	frame = vs.read()
	frame = imutils.resize(frame, width=600)

	# blob 이미지로 변환
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

	# blob 이미지를 통해 OpenCV의 딥러닝 기반 얼굴 탐지기 이용하여 탐지 및 예측 진행
	net.setInput(blob)
	detections = net.forward()

	# 탐지 반복
	for i in range(0, detections.shape[2]):
		# 예측과 관련된 신뢰도(확률)를 추출
		confidence = detections[0, 0, i, 2]

		# 최소 확률 감지 임계값과 비교하여 계산된 확률이 최소 확률보다 큰지 확인
		if confidence > args["confidence"]:
			# 얼굴 경계 상자의 (x, y) 좌표를 계산하고 얼굴 ROI 추출
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# 감지된 경계 상자가 프레임의 치수를 벗어나지 않도록 주의
			startX = max(0, startX)
			startY = max(0, startY)
			endX = min(w, endX)
			endY = min(h, endY)

			# 얼굴 ROI를 추출한 다음 훈련 데이터와 정확히 동일한 방식으로 선행 처리
			face = frame[startY:endY, startX:endX]
			face = cv2.resize(face, (32, 32))
			face = face.astype("float") / 255.0
			face = img_to_array(face)
			face = np.expand_dims(face, axis=0)

			# 훈련된 진짜 얼굴 탐지기 모델을 통해 얼굴 ROI를 전달하여 얼굴이 진짜인지 가짜인지 확인
			preds = model.predict(face)[0]
			j = np.argmax(preds)
			label = le.classes_[j]

			# 프레임의 경계 상자와 레이블 그린다.
			label = "{}: {:.4f}".format(label, preds[j])
			cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

		if confidence > args["confidence"]:
			# 얼굴 경계 상자의 (x,y) 좌표 계산
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# 얼굴 ROI 추출
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# 얼굴 너비와 높이가 충분히 큰지 확인
			if fW < 20 or fH < 20:
				print("ROI 사이즈 : ", fH, " ", fW)
				continue

			# 얼굴 ROI에 대한 blob을 구성한 다음 얼굴 임베딩 모델을 통해 blob을 전달하여 얼굴의 128-D 벡터 생성
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			# 벡터를 SVM 인식기 모델을 통해 전달
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# 얼굴을 인식하기 위해 분류를 수행 (가장 높은 확률 지수를 취하고 이름을 찾기 위해 레이블 인코더 색인)
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le2.classes_[j]

			# $$$$ 추가 $$$$
			# 가장 인식이 많이 된 사람 ROI 사이즈 기준 동영상 실행시 누적해서 계속 값을 쌓고,
			# 영상이 종료되면 MAX 누적값에 해당하는 이름을 출력한다.
			#print(le.classes_)
			print(name)
			if name != 'unknown' :
				#print(name ," fH: ", fH)
				put(people, name, fH)



			# 관련 확률과 함께 얼굴의 경계 상자 그림
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 30 if startY - 30 > 10 else startY + 30
			cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
	# 출력 프레임을 표시
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# 'q' 키가 입력되면 루프 탈출
	if key == ord("q"):
		break

# Cleaning
cv2.destroyAllWindows()
vs.stop()

#저장된 people 리스트에서 가장 큰 ROI 박스가 누적된 사람을 뽑아 출력한다.
dicArr = sorted(people.items(), key=lambda x: x[1], reverse=True)
print("사용자는 "+ dicArr[0][0] + " 입니다.")
print(dicArr[:50])

