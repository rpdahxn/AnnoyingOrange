import cv2
import dlib
from imutils import face_utils, resize
import numpy as np

annoying_img = cv2.imread('/Users/dahyeon/projects/MEMES-GENERATOR/images/peach.jpeg')
annoying_img = cv2.resize(annoying_img, dsize = (512, 512))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/Users/dahyeon/Downloads/shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)  # 0은 웹캠을 사용한다.

while cap.isOpened():
    ret, img = cap.read()

    # 프레임이 더 없으면 반복문을 빠져나온다.
    if not ret:
         break
    
    faces = detector(img)

    result = annoying_img.copy()

    if len(faces) > 0:
         face = faces[0]

         x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
         face_img = img[y1-100:y2+100, x1-100:x2+100].copy()

         shape = predictor(img, face)
         # shape에 68개의 점의 정보가 담긴다.
         shape = face_utils.shape_to_np(shape)

         for p in shape:
            cv2.circle(face_img, center=(p[0] - x1, p[1] - y1), radius=2, color=255, thickness=-1)

        #  cv2.imshow('face', face_img)
         
         # eyes
         le_x1 = shape[36, 0]
         le_y1 = shape[37, 1]
         le_x2 = shape[39, 0]
         le_y2 = shape[41, 1]
         le_margin = int((le_x2 - le_x1) * 0.18)
    
         re_x1 = shape[42, 0]
         re_y1 = shape[43, 1]
         re_x2 = shape[45, 0]
         re_y2 = shape[47, 1]
         re_margin = int((re_x2 - re_x1) * 0.18)

         left_eye_img = img[le_y1-le_margin:le_y2+le_margin, le_x1-le_margin:le_x2+le_margin].copy()
         right_eye_img = img[re_y1-re_margin:re_y2+re_margin, re_x1-re_margin:re_x2+re_margin].copy()

         left_eye_img = resize(left_eye_img, width=100)
         right_eye_img = resize(right_eye_img, width=100)

         # seamlessClone : seamless하게 합성해준다.
         result = cv2.seamlessClone(
            left_eye_img,
            result,  # 오렌지 이미지인 result에 합성
            np.full(left_eye_img.shape[:2], 255, left_eye_img.dtype),
            (100, 200),
            cv2.NORMAL_CLONE
        )

         result = cv2.seamlessClone(
            right_eye_img,
            result,
            np.full(right_eye_img.shape[:2], 255, right_eye_img.dtype),
            (250, 200),
            cv2.NORMAL_CLONE
        )
         
         # mouth
         mouth_x1 = shape[48, 0]
         mouth_y1 = shape[50, 1]
         mouth_x2 = shape[54, 0]
         mouth_y2 = shape[57, 1]
         mouth_margin = int((mouth_x2 - mouth_x1) * 0.1)

         mouth_img = img[mouth_y1-mouth_margin:mouth_y2+mouth_margin, mouth_x1-mouth_margin:mouth_x2+mouth_margin].copy()

         mouth_img = resize(mouth_img, width=250)

         result = cv2.seamlessClone(
            mouth_img,
            result,
            np.full(mouth_img.shape[:2], 255, mouth_img.dtype),
            (180, 320),
            cv2.NORMAL_CLONE
        )
         
        #  cv2.imshow('left', left_eye_img)
        #  cv2.imshow('right', right_eye_img)
        #  cv2.imshow('mouth', mouth_img)

         cv2.imshow('result', result)


    if cv2.waitKey(1) == ord('q'):
        break





