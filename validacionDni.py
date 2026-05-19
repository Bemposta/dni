import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import json
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
from deepface import DeepFace

'''************************************************************************************************
## Importante para la instacion de TensorFlow:
  * https://developer.nvidia.com/cuda-downloads
  * pip install tensorflow[and-cuda]
  * export CUDA_HOME="/usr/local/cuda"
  * export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
  * export LD_LIBRARY_PATH="/usr/local/cuda/lib64"
  * python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
  * sudo apt install nvidia-cuda-toolkit
  * nvcc --version
  * python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
  * --> Importante: "reboot"
  
************************************************************************************************'''


#************************************************************************************************
# Detector y aislador de DNI en imagen
#************************************************************************************************
class DetectorDni:
    widthDni = 640
    heightDni = 400
    heightImgFrame = 960

    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("===================================================================")
        print(f"Trabajando con: {device}")
        print("===================================================================")

        # 1. Cargar el modelo de pose y moverlo a la CPU
        modeloRedpose = 'midvall_ss_px960_e40.pt'
        self.modelpose = YOLO(modeloRedpose).to(device)
        modeloReddetect = 'dni_sn_px640_e200.pt'
        self.modeldetect = YOLO(modeloReddetect).to(device)

        self.imgFrame = None
        self.imagenDNI = None
        self.esquinas = None
        self.error = ""
        self.claves = None
        self.imagenDniAnnotated = None
        self.clavesSimple = None

    #************************************************************************************************
    def recortarDNI(self):
        rect = np.array(self.esquinas, dtype="float32")
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))
        dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        self.imagenDNI = cv2.warpPerspective(self.imgFrame, M, (maxWidth, maxHeight))
        return True

    #************************************************************************************************
    def detectDNI(self):
        results = self.modelpose.predict(source=self.imgFrame, imgsz=self.heightImgFrame, rect=True, verbose=False)
        if len(results) > 1:
            print("Mas de una imagen en buffer. ABORTADO")
            return False
        result = results[0]
        if result.keypoints is None:
            print("No se ha detectado DNI. ABORTADO")
            return False
        kpts = result.keypoints.xy.cpu().numpy()
        confs = result.keypoints.conf.cpu().numpy()
        if len(kpts) != 1:
            print(f"Detectados {len(kpts)} DNI's en la imagen. ABORTADO")
            return False
        puntos_obj = kpts[0]
        conf_obj = confs[0]
        # Filtramos solo los puntos con buena confianza para evitar líneas erróneas
        self.esquinas = [p.astype(int) for i, p in enumerate(puntos_obj) if conf_obj[i] > 0.4]
        if len(self.esquinas) != 4:
            print("Error en la deteccion de 4 esquinas . ABORTADO")
            return False
        return True

    #************************************************************************************************
    def showKeypoints(self):
        frame = self.imgFrame.copy()
        pts = np.array(self.esquinas, np.int32)
        cv2.polylines(frame, pts=[pts], isClosed=True, color=(0, 255, 0), thickness=4)
        for p in pts:
            cv2.circle(frame, center=p, radius=10, color=(0, 0, 255), thickness=-1)
        cv2.namedWindow(winname="Keypoints", flags=cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO )
        h, w = frame.shape[:2]
        ancho = int((float(self.heightImgFrame)/h)*w)
        cv2.resizeWindow(winname="Keypoints", width=ancho, height=self.heightImgFrame)
        cv2.imshow(winname="Keypoints", mat=frame)

    #************************************************************************************************
    def buscarClavesEnDNI(self):
        results = self.modeldetect.predict(self.imagenDNI, imgsz=self.widthDni, rect=True, verbose=False)
        if len(results) > 1:
            print("Mas de una imagen en buffer DNI. ABORTADO")
            return False
        result = results[0]
        self.claves = result.to_json()
        parsed_json = json.loads(self.claves)
        self.clavesSimple = [(d["name"], d["confidence"]) for d in parsed_json]
        self.imagenDniAnnotated = result.plot()
        return True

    #************************************************************************************************
    def showClavesEnDNI(self):
        cv2.namedWindow(winname="Deteccion", flags=cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO )
        h, w = self.imagenDniAnnotated.shape[:2]
        cv2.resizeWindow(winname="Deteccion", width=self.widthDni, height=self.heightDni)
        cv2.imshow(winname="Deteccion", mat=self.imagenDniAnnotated)

    # ************************************************************************************************
    def procesarImagen(self, frame):
        self.imagenDNI = None
        self.esquinas = None
        self.error = ""
        self.claves = None
        self.imagenDniAnnotated = None
        self.imgFrame = frame
        ok = self.detectDNI()
        if not ok:
            print("Error en esquinas. TERMINADO")
            return False
        ok = self.recortarDNI()
        if not ok:
            print("DNI no detectado. TERMINADO")
            return False
        ok = self.buscarClavesEnDNI()
        if not ok:
            print("No encontradas claves en DNI. TERMINADO")
            return False
        return True

#************************************************************************************************
# Comparador de identidad de la imagen con Selfie
#************************************************************************************************
class DniSelfie:

    def __init__(self, net=2):
        # 1. Cargar el modelo de pose y moverlo a la CPU
        modelosRedpname = ['ArcFace', 'Facenet', 'Facenet512', 'VGG-Face', 'SFace', 'DeepFace', 'OpenFace']
        self.modeloRedpname = modelosRedpname[net]
        self.imgSelfie = None
        self.imagenDNI = None
        self.coincidencia = None
        self.coparacionImg = None

        # Definir dimensiones (puedes usar 160x160 o 224x224 que son comunes en DeepFace)
        width, height = 224, 224
        ruido1 = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        ruido2 = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        try:
            DeepFace.verify(
                img1_path = ruido1,
                img2_path = ruido2,
                model_name = self.modeloRedpname,
                enforce_detection = False
            )
        except Exception:
            print("", end="")
        return

    # ************************************************************************************************
    def veriryFace(self, imgDNI, imgSelfie):
        self.imagenSelfie = imgSelfie
        self.imagenDNI = imgDNI
        try:
            self.coincidencia = DeepFace.verify(
                img1_path = self.imagenDNI,
                img2_path = self.imagenSelfie,
                model_name = self.modeloRedpname,
                enforce_detection = True
            )
        except ValueError as e:
            print(f"Error: No se detectó ningún rostro. Por favor, mire a la cámara. {e}")
            return False
        except Exception as e:
            print(f"Error inesperado: {e}")
            return False
        return self.coincidencia["verified"]

    # ************************************************************************************************
    def verirySpoofing(self, imgSelfie):

        try:
            # Analizamos cada frame manualmente para aplicar anti_spoofing
            results = DeepFace.extract_faces(
                img_path=imgSelfie,
                detector_backend='retinaface',
                anti_spoofing=True,
                enforce_detection=True
            )

            for face in results:
                x, y, w, h = [face["facial_area"][k] for k in ('x', 'y', 'w', 'h')]
                left_eye, right_eye = [face["facial_area"][k] for k in ('left_eye', 'right_eye')]
                is_real = face.get("is_real", False)
                color = (0, 255, 0) if is_real else (0, 0, 255)  # Verde si es real, Rojo si es falso
                label = "REAL" if is_real else "FAKE / SPOOF"

                cv2.rectangle(imgSelfie, (x, y), (x + w, y + h), color, 2)
                cv2.circle(imgSelfie, left_eye, 5, color, 2)
                cv2.circle(imgSelfie, right_eye, 5, color, 2)
                cv2.putText(imgSelfie, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except ValueError as e:
            print(f"Error: No se detectó ningún rostro. Por favor, mire a la cámara. {e}")
            return False, None
        except Exception as e:
            print(f"Error inesperado: {e}")
            return False, None

        print("Spoofing:", results[0]['is_real'], results[0]['confidence'], results[0]['antispoof_score'])

        if len(results) == 1:
            return True, imgSelfie

        return False, imgSelfie

    # ************************************************************************************************
    def createCoparacionImg(self):
        if self.coincidencia is None:
            print("No hay comparacion. TERMINADO")
            return
        faces = []
        # 2. Extraer y recortar cara
        area1 = self.coincidencia['facial_areas']['img1']
        area2 = self.coincidencia['facial_areas']['img2']

        x, y, w, h = area1['x'], area1['y'], area1['w'], area1['h']
        recorte = self.imagenDNI[y:y + h, x:x + w]
        faces.append(recorte)
        x, y, w, h = area2['x'], area2['y'], area2['w'], area2['h']
        recorte = self.imagenSelfie[y:y + h, x:x + w]
        faces.append(recorte)

        h_target = 300
        resized_faces = []
        for f in faces:
            aspect_ratio = f.shape[1] / f.shape[0]
            w_target = int(h_target * aspect_ratio)
            resized_faces.append(cv2.resize(f, (w_target, h_target)))
        self.coparacionImg = np.hstack(resized_faces)

        if self.coincidencia["verified"]:
            texto = "True"
            color = (0, 255, 0)  # Verde en BGR
        else:
            texto = "False"
            color = (0, 0, 255)  # Rojo en BGRR

        posicion = (320, 60)
        fuente = cv2.FONT_HERSHEY_SIMPLEX
        escala = 1.5
        grosor = 3
        cv2.putText(self.coparacionImg, texto, posicion, fuente, escala, color, grosor, cv2.LINE_AA)

        return

    # ************************************************************************************************
    def showCoparacion(self):
        if self.coparacionImg is None:
            self.createCoparacionImg()
        cv2.imshow('Comparacion de Rostros', self.coparacionImg)
        return

#************************************************************************************************
# Programa Principal.
#************************************************************************************************
# ************************************************************************************************
def procesar_app(imgDNI, imgSelfie):
    print("*********************************************************************************")
    print(f"Procesando: {imgDNI}")
    detectorDni = DetectorDni()
    fotoVerify = DniSelfie()
    frame = cv2.imread(imgDNI)
    if frame is None:
        print(f"Error: No se pudo leer {imgDNI}")
        return
    selfie = cv2.imread(imgSelfie)
    if frame is None:
        print(f"Error: No se pudo leer {imgSelfie}")
        return
    ok = detectorDni.procesarImagen(frame)
    if not ok:
        print(f"Algun error: {detectorDni.error}")
        return
    #print(detectorDni.clavesSimple)
    print(detectorDni.claves)
    ok = fotoVerify.veriryFace(detectorDni.imagenDNI, selfie)
    if not ok:
        print("No hay concidencia!!")
    else:
        print("coincidencia!! :", fotoVerify.coincidencia)

# ************************************************************************************************  
def procesar_file(imgDNI, imgSelfie):
    print("*********************************************************************************")
    print(f"Procesando: {imgDNI}")
    detectorDni = DetectorDni()
    fotoVerify = DniSelfie()
    frame = cv2.imread(imgDNI)
    if frame is None:
        print(f"Error: No se pudo leer {imgDNI}")
        return
    selfie = cv2.imread(imgSelfie)
    if frame is None:
        print(f"Error: No se pudo leer {imgSelfie}")
        return
    ok = detectorDni.procesarImagen(frame)
    if not ok:
        print(f"Algun error: {detectorDni.error}")
        return
    print(detectorDni.clavesSimple)
    #print(detectorDni.claves)
    detectorDni.showKeypoints()
    detectorDni.showClavesEnDNI()
    ok = fotoVerify.veriryFace(detectorDni.imagenDNI, selfie)
    if not ok:
        print("No hay concidencia!!")
    else:
        print("coincidencia!! :", fotoVerify.coincidencia)
    fotoVerify.showCoparacion()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ************************************************************************************************
def procesar_directorio(ruta_carpeta):
    detectorDni = DetectorDni()
    path_dir = Path(ruta_carpeta)
    extensiones = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']
    archivos_imagen = []
    for ext in extensiones:
        archivos_imagen.extend(path_dir.glob(ext))
    print(f"Se encontraron {len(archivos_imagen)} imágenes.")
    for ruta_archivo in archivos_imagen:
        print("*********************************************************************************")
        print(f"Procesando: {ruta_archivo.name}")
        frame = cv2.imread(str(ruta_archivo))
        if frame is not None:
            ok = detectorDni.procesarImagen(frame)
            if not ok:
                print(f"Algun error: {detectorDni.error}")
                continue
            print(detectorDni.clavesSimple)
            detectorDni.showKeypoints()
            detectorDni.showClavesEnDNI()
            cv2.waitKey(1000)
        else:
            print(f"Error: No se pudo leer {ruta_archivo}")
    cv2.destroyAllWindows()

# ************************************************************************************************
# --- EJECUCIÓN ---
# ************************************************************************************************
if __name__ == "__main__":
    procesar = "app"
    if procesar == "Directorio":
        mi_ruta = "C:/DataSets/Test"  # Cambia esto por tu ruta
        procesar_directorio(mi_ruta)
    if procesar == "File":
        imagenDNI = "20_0289_.jpg"
        imgenSelfie = "mrBean4.jpg"
        procesar_file(imagenDNI, imgenSelfie)
    if procesar == "app":
        imagenDNI = "20_0289_.jpg"
        imgenSelfie = "zapatero2.jpg"
        procesar_app(imagenDNI, imgenSelfie)
    exit()

