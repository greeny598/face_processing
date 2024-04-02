# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 22:49:16 2024

@author: Professional
"""
import cv2
import numpy as np
import os

from face_processor import FaceProcessor
import matplotlib.pyplot as plt

fig = plt.figure(constrained_layout=True)
fig.set_facecolor('w')
ax = fig.add_subplot(
    projection='3d'
    )
ax.elev = -2
ax.azim = 90
ax.set_box_aspect(None, zoom=1)
ax.set_facecolor('w')
# ax.set_aspect(aspect="equal")# не заработало под linux
ax.set_aspect(aspect="auto")
####--------------------------------------------------------------------------


cv2.namedWindow('Result', cv2.WINDOW_NORMAL)

# Установка желаемых размеров окна
cv2.resizeWindow('Result', 300, 300)  # Задайте нужные размеры

# Путь к обрабатываемому видеофайлу
video_path = 'D:/Work/Polygraph/lie/'  # Укажите путь к папке


###----------------------------------------------------------------------------

files = os.listdir(video_path)

name_without_ext = os.path.splitext(os.path.basename(video_path))[0]




##при использовании 478 лендмарков лица получаем 1434 координаты
##автоматизировать в будущем
coord_vectors=np.empty([1, 1434])





cap = cv2.VideoCapture(video_path)
processor = FaceProcessor()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("I`gnoring empty camera frame.")
        break

    # Обработка изображения с помощью класса FaceMeshProcessor
    cropped_face = processor.detect_and_crop_face(frame)
    landmarks = processor.detect_landmarks(cropped_face)
    if landmarks:
        annotated_image = processor.draw_landmarks(landmarks, cropped_face)
        landmarks_coordinates = processor.get_landmarks_coordinates(landmarks)
    else:
        annotated_image = cropped_face

    # Воспроизведение обработанного видео
    cv2.imshow('Result', annotated_image)
    ##вызов функции отрисовки лендмарков в 3D координатах
    processor.render_face_3d(landmarks_coordinates, ax)
    
    ##дописывем строку координат в буфер
    vector = landmarks_coordinates.reshape(1,-1)
    coord_vectors = np.vstack((coord_vectors, vector))
    
    if cv2.waitKey(5) & 0xFF == 27:  # Закрыть окно по нажатию ESC
        break
    

cap.release()
cv2.destroyAllWindows()

np.savetxt(f"./tr_csv/{name_without_ext}.csv", coord_vectors, delimiter=",")
