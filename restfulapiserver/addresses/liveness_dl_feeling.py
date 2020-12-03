# /usr/bin/python3
import cv2
import numpy as np
import sys
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import tensorflow as tf

from model import predict, image_to_tensor, deepnn
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
from keras.models import load_model
from keras.preprocessing.image import img_to_array

CASC_PATH = 'data/haarcascade_files/haarcascade_frontalface_default.xml'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']


#  ------------------------
import os
DIR = 'output'



def format_image(image):
  if len(image.shape) > 2 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = cascade_classifier.detectMultiScale(
    image,
    scaleFactor = 1.3,
    minNeighbors = 5
  )
  # None is no face found in image
  if not len(faces) > 0:
    return None, None
  max_are_face = faces[0]
  for face in faces:
    if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
      max_are_face = face
  # face to image
  face_coor =  max_are_face
  image = image[face_coor[1]:(face_coor[1] + face_coor[2]), face_coor[0]:(face_coor[0] + face_coor[3])]
  # Resize image to network size
  try:
    image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC)
  except Exception:
    print("[+} Problem during resize")
    return None, None
  return  image, face_coor

def face_dect(image):
  """
  Detecting faces in image
  :param image:
  :return:  the coordinate of max face
  """
  if len(image.shape) > 2 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = cascade_classifier.detectMultiScale(
    image,
    scaleFactor = 1.3,
    minNeighbors = 5
  )
  if not len(faces) > 0:
    return None
  max_face = faces[0]
  for face in faces:
    if face[2] * face[3] > max_face[2] * max_face[3]:
      max_face = face
  face_image = image[max_face[1]:(max_face[1] + max_face[2]), max_face[0]:(max_face[0] + max_face[3])]
  try:
    image = cv2.resize(face_image, (48, 48), interpolation=cv2.INTER_CUBIC) / 255.
  except Exception:
    print("[+} Problem during resize")
    return None
  return face_image

def resize_image(image, size):
  try:
    image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC) / 255.
  except Exception:
    print("+} Problem during resize")
    return None
  return image

def draw_emotion():
  pass

people = {}
real_or_fake = {}
emotions = {}

def put(people, name, value):
    if name not in people:
        people[name] = value
    else:
        people[name] += value


def put_rf(real_or_fake, name, value):
    if name not in real_or_fake:
        real_or_fake[name] = value
    else:
        real_or_fake[name] += value;


def put_feeling(emotions, feeling, value):
  if feeling not in emotions:
    emotions[feeling] = value
  else:
    emotions[feeling] += value



def demo(DETECTOR, MODEL, LE, EMBEDDING_MODEL, RECOGNIZER, LE2, INPUT, OUTPUT, SKIP, CONFIDENCE, modelPath, showBox=False) :
  face_x = tf.placeholder(tf.float32, [None, 2304])
  y_conv = deepnn(face_x)
  probs = tf.nn.softmax(y_conv)

  saver = tf.train.Saver()
  ckpt = tf.train.get_checkpoint_state(modelPath)
  sess = tf.Session()
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('Restore model sucsses!!\nNOTE: Press SPACE on keyboard to capture face.')

  feelings_faces = []
  for index, emotion in enumerate(EMOTIONS):
    feelings_faces.append(cv2.imread('./data/emojis/' + emotion + '.png', -1))
  # video_captor = cv2.VideoCapture(0)
  # ---------------------------------------------------------------------------------------------
  # 얼굴 탐지기 로딩
  # 얼굴을 감지하기 위해 OpenCV에서 제공하는 사전 훈련된 Caffe 딥러닝 모델
  print("[INFO] loading face detector...")
  protoPath = os.path.sep.join([DETECTOR, "deploy.prototxt"])
  modelPath = os.path.sep.join([DETECTOR, "res10_300x300_ssd_iter_140000.caffemodel"])
  detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

  # 진짜 얼굴 탐지 모델 및 레이블 로딩
  print("[INFO] loading liveness detector...")

  # file_name = os.path.dirname(__file__) + '\\my_folder\\my_model_1.h5'
  # f = h5py.File(file_name)

  model = load_model(MODEL)
  le = pickle.loads(open(LE, "rb").read())

  # 얼굴 인식기 로딩 (128-D 얼굴 인식을 계산하기 위해 사전 훈련된 Torch DL 모델)
  print("[INFO] loading face recognizer...")
  embedder = cv2.dnn.readNetFromTorch(EMBEDDING_MODEL)

  # 레이블 인코더와 함께 실제 얼굴 인식 모델 로딩 (Linear SVM얼굴 인식모델)
  recognizer = pickle.loads(open(RECOGNIZER, "rb").read())
  le2 = pickle.loads(open(LE2, "rb").read())

  #  ++ 추가
  # 비디오 파일 스트림 초기화

  vs = cv2.VideoCapture(INPUT)

  list = os.listdir(DIR)
  file_count = len(list)
  read = file_count + 1
  saved = file_count + 1
  # ---------------------------------------------------------------------------------------------

  emoji_face = []
  result = None

  # ---------------------------------------------------------------------------------------------
  list = os.listdir(DIR)
  file_count = len(list)
  read = file_count + 1
  saved = file_count + 1
  # ---------------------------------------------------------------------------------------------

  while True:
    # 파일에서 비디오 스트림 프레임 입력
    (grabbed, frame) = vs.read()

    # 더이상 프레임이 없으면 루프 탈출
    if not grabbed:
      break

    # 프레임수 증가
    read += 1


    # ㅁㅁㅁ Skip aaa
    if read %  SKIP != 0:
      continue
    detected_face, face_coor = format_image(frame)

    # 프레임에서 blob 구성
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # 입력된 이미지에서 얼굴을 인식하기 위해 OpenCV의 딥러닝 기반 얼굴 탐지기 이용
    detector.setInput(blob)
    detections = detector.forward()

    # 적어도 하나의 얼굴이 발견되었는지 확인
    if len(detections) > 0:
      # 각 이미지가 하나의 얼굴만을 가지고 있다고 가정하고, 가장 큰 확률을 가진 경계 상자를 찾음
      i = np.argmax(detections[0, 0, :, 2])
      confidence2 = detections[0, 0, i, 2]

      # 확률이 가장 큰 탐지는 최소 확률 테스트를 의미
      if confidence2 > CONFIDENCE :
        # 얼굴 경계 상자의 (x,y) 좌표 계산하고 얼굴 ROI 추출
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        face = frame[startY:endY, startX:endX]

        # # 프레임 쓰기
        # p = os.path.sep.join([args["output"], "{}.png".format(saved)])
        # cv2.imwrite(p, frame)
        # saved += 1
        # print("[INFO] saved {} to disk".format(p))

      # 최소 확률 감지 임계값과 비교하여 계산된 확률이 최소 확률보다 큰지 확인
      if confidence2 > CONFIDENCE :
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
        # print(j)
        label = le.classes_[j]
        print(label)
        put_rf(real_or_fake, label, preds[j])

        # 프레임의 경계 상자와 레이블 그린다.
        label = "{}: {:.4f}".format(label, preds[j])
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

    if confidence2 > CONFIDENCE :
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

      # 얼 굴을 인식하기 위해 분류를 수행 (가장 높은 확률 지수를 취하고 이름을 찾기 위해 레이블 인코더 색인)
      preds = recognizer.predict_proba(vec)[0]
      j = np.argmax(preds)
      proba = preds[j]
      name = le2.classes_[j]

      # $$$$ 추가 $$$$
      # 가장 인식이 많이 된 사람 ROI 사이즈 기준 동영상 실행시 누적해서 계속 값을 쌓고,
      # 영상이 종료되면 MAX 누적값에 해당하는 이름을 출력한다.

      print('name : ' ,name)
      put(people, name, fH)
      if name not in 'real':
        if name not in 'fake':
          print(i)
          put(people, name, fH)

      # 관련 확률과 함께 얼굴의 경계 상자 그림
      text = "{}: {:.2f}%".format(name, proba * 100)
      y = startY - 30 if startY - 30 > 10 else startY + 30
      cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
      cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    #     -------------------------------------------------------------------------------------
    if showBox:
      if face_coor is not None:
        [x, y, w, h] = face_coor
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if detected_face is not None:
      cv2.imwrite('a.jpg', detected_face)
      tensor = image_to_tensor(detected_face)
      result = sess.run(probs, feed_dict={face_x: tensor})

      print('result ', result)

      # 감정 딕셔너리 추가
      for i in range(len(EMOTIONS)):
        put_feeling(emotions, EMOTIONS[i], int(result[0][i] * 100))

    if result is not None:
      for index, emotion in enumerate(EMOTIONS):
        cv2.putText(frame, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
        cv2.rectangle(frame, (130, index * 20 + 10), (130 + int(result[0][index] * 100), (index + 1) * 20 + 4),
                      (255, 0, 0), -1)
        emoji_face = feelings_faces[np.argmax(result[0])]

      for c in range(0, 3):
        frame[200:320, 10:130, c] = emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0) + frame[200:320, 10:130, c] * (
                1.0 - emoji_face[:, :, 3] / 255.0)
      cv2.imshow('face', frame)
      # 프레임 쓰기
      p = os.path.sep.join([OUTPUT, "{}.png".format(saved)])
      cv2.imwrite(p, frame)
      saved += 1
      print("[INFO] saved {} to disk".format(p))
      print('le2.classes_ : ', le2.classes_)

  dicArr_REALFAKE = sorted(real_or_fake.items(), key=lambda x: x[1], reverse=True)
  dicArr_WHO = sorted(people.items(), key=lambda x: x[1], reverse=True)
  dicArr_feeling = sorted(emotions.items(), key=lambda x: x[1], reverse=True)

  # spoofing = dicArr_REALFAKE[0][0]
  who = dicArr_WHO[0][0]

  print('dicArr_REALFAKE : ', dicArr_REALFAKE)
  print('dicArr_WHO : ', dicArr_WHO)
  print(dicArr_feeling)

  print(dicArr_REALFAKE[0][0] , " :  ", dicArr_WHO[0][0], "  :  ",  dicArr_feeling[:3] )



