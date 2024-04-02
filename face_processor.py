import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt


class FaceProcessor:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, ##для выбора детекции на близком расстоянии. Важно!
            min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            ##режим уточненных лендмарков, нужен для отображения зрачков
            ##дает 478 лендмарков вместо 468
            refine_landmarks=True,
            min_tracking_confidence=0.6)
        
        
    def detect_and_crop_face(self, image):
        results = self.face_detection.process(image)
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
    
                # Рассчитываем поля вокруг лица, равные половине размеров лица
                padding_x = int(w*0.8)
                padding_y = int(h*0.8)
    
                # Расширяем область обрезки, добавляя поля
                x = max(x - padding_x, 0)
                y = max(y - padding_y, 0)
                w = min(w + 2 * padding_x, iw - x)
                h = min(h + 2 * padding_y, ih - y)
    
                # Возвращаем обрезанный до лица кадр с дополнительными полями
                return image[y:y+h, x:x+w]
        return image  # Возвращаем исходный кадр, если лицо не обнаружено


    def detect_landmarks(self, image):
        # Конвертация изображения из BGR в RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Обнаружение лица и распознавание лендмарков
        results = self.face_mesh.process(image)

        # Конвертация изображения обратно в BGR для отображения
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        landmarks = None

        # обнаружение лендмарков на лице
        if results.multi_face_landmarks:
            landmarks = [face_landmarks for face_landmarks in results.multi_face_landmarks]
        else:
            print('can not find face landmarks in a frame')
        return landmarks
            
    def draw_landmarks(self, landmarks, image):
        for face_landmarks in landmarks:
            self.mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None, 
                # connection_drawing_spec=self.mp_drawing_styles
                # .get_default_face_mesh_tesselation_style())
                connection_drawing_spec = self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0), 
                    thickness=1,
                    circle_radius = 1))
            # ### отображение контуров губ и глаз
            # self.mp_drawing.draw_landmarks(
            #     image=image,
            #     landmark_list=face_landmarks,
            #     connections=self.mp_face_mesh.FACEMESH_CONTOURS,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
            # ### отображение зрачков
            # self.mp_drawing.draw_landmarks(
            #     image=image,
            #     landmark_list=face_landmarks,
            #     connections=self.mp_face_mesh.FACEMESH_IRISES,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style())
        return image
    
    def render_face_3d(self,coords, ax):
        plt.cla()
        ax.scatter(-coords[:, 0], coords[:, 2], -coords[:, 1], c=coords[:, 2],
                        cmap="ocean", clip_on=False, 
                        vmax=2*coords[:, 2].max(),
                        marker = 'o',
                        ##marker size
                        s = 4)
        ax.axis("off")
        plt.draw()
    
    
    def get_landmarks_coordinates(self, landmarks):
        landmarks_list = []
        for face_landmarks in landmarks:
            landmarks = []
            for landmark in face_landmarks.landmark:
                landmarks.append((landmark.x, landmark.y, landmark.z))
            landmarks_list.append(landmarks)
            landmarks_coordinates = np.asarray(landmarks_list).reshape(-1,3)
            return landmarks_coordinates
        
        