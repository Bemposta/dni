from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import FileResponse
import cv2
from validacionDni import DetectorDni, DniSelfie
import numpy as np

"""
Comando para ejecutarlo:
     uvicorn validacion_api:app --reload
"""

app = FastAPI()
detectorDni = DetectorDni()
fotoVerify = DniSelfie()

# Ruta principal GET que devuelve el archivo HTML
@app.get("/", response_class=FileResponse)
async def raiz():
    # Buscamos el archivo index.html en la misma carpeta
    ruta_html = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(ruta_html)

@app.post("/imagenes/")
async def procesar_imagenes(
    fotoDni: UploadFile = File(...),
    imgSlf: UploadFile = File(...)
):

    fotoDni64 = await fotoDni.read()
    imgSlf64 = await imgSlf.read()

    fotoDniNpi = np.frombuffer(fotoDni64, np.uint8)
    imgSlfNpi = np.frombuffer(imgSlf64, np.uint8)

    fotoDniCv2 = cv2.imdecode(fotoDniNpi, cv2.IMREAD_COLOR)
    imgSlfCv2 = cv2.imdecode(imgSlfNpi, cv2.IMREAD_COLOR)

    if fotoDniCv2 is None or imgSlfCv2 is None:
        return {"error": "El archivo enviado no es una imagen válida o está corrupto"}
    # 2. Inferencia con YOLO
    ok = detectorDni.procesarImagen(fotoDniCv2)
    if not ok:
        return {"error": "No se ha detecctado DNI"}

    print(detectorDni.clavesSimple)

    ok = fotoVerify.veriryFace(detectorDni.imagenDNI, imgSlfCv2)
    if not ok:
        print("error, Selfie y DNI no coincidentes")
    else:
        print("coincidencia!! :", fotoVerify.coincidencia)
    fotoVerify.createCoparacionImg()

    # 3. Convertir el array de imagen (BGR) a formato compatible con web (RGB -> PNG/JPG)
    imagenDni_reescalada = cv2.resize(detectorDni.imagenDniAnnotated, (640, 400), interpolation=cv2.INTER_AREA)
    imagenSelfie_reescalada = cv2.resize(fotoVerify.coparacionImg, (640, 300), interpolation=cv2.INTER_AREA)
    resultado = cv2.vconcat([imagenDni_reescalada, imagenSelfie_reescalada])
    exito, buffer = cv2.imencode('.webp', resultado)
    if not exito:
        return {"error": "No se pudo codificar la imagen"}
    img_bytes = buffer.tobytes()

    print("Terminado correctamente.")
    # 4. Devolver la imagen directamente al navegador
    return Response(content=img_bytes, media_type="image/webp")
