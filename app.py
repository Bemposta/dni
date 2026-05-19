from fastapi import FastAPI, File, UploadFile, Response, HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io
import cv2
from validacionDni import DetectorDni, DniSelfie
import numpy as np
import os
import json

"""
Comando para ejecutarlo:
     uvicorn validacion_api:app --reload
"""

app = FastAPI()
detectorDni = DetectorDni()
fotoVerify = DniSelfie()

class ResultadoDNI(BaseModel):
    status: bool
    code: str
    clases: list

# Ruta principal GET que devuelve el archivo HTML
@app.get("/", response_class=FileResponse)
async def raiz():
    # Buscamos el archivo index.html en la misma carpeta
    ruta_html = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(ruta_html)

@app.post("/procesar/")
async def procesar_imagenes( fotoDni: UploadFile = File(...), imgSlf: UploadFile = File(...) ):
    fotoDni64 = await fotoDni.read()
    imgSlf64 = await imgSlf.read()

    fotoDniNpi = np.frombuffer(fotoDni64, np.uint8)
    imgSlfNpi = np.frombuffer(imgSlf64, np.uint8)

    fotoDniCv2 = cv2.imdecode(fotoDniNpi, cv2.IMREAD_COLOR)
    imgSlfCv2 = cv2.imdecode(imgSlfNpi, cv2.IMREAD_COLOR)

    if fotoDniCv2 is None or imgSlfCv2 is None:
        return ResultadoDNI(
            status = False,
            code = "El archivo enviado no es una imagen válida o está corrupto.",
            clases = None
        )

    ok = detectorDni.procesarImagen(fotoDniCv2)
    if not ok:
        return ResultadoDNI(
            status = False,
            code = "DNI no encontrado.",
            clases = None
        )

    ok = fotoVerify.veriryFace(detectorDni.imagenDNI, imgSlfCv2)
    if not ok:
        return ResultadoDNI(
            status = False,
            code = "DNI y Selfie no coincidentes.",
            clases = json.loads(detectorDni.claves)
        )

    return ResultadoDNI(
        status = True,
        code = "DNI y Selfie coincidentes.",
        clases = json.loads(detectorDni.claves)
    )

# --- Endpoint de imagen: sirve cada imagen por separado ---
@app.get("/imagen/imgdni/")
def get_imagen_dni() -> StreamingResponse:
    imagenDni_reescalada = cv2.resize(detectorDni.imagenDniAnnotated, (640, 400), interpolation=cv2.INTER_AREA)
    exito, buffer = cv2.imencode('.webp', imagenDni_reescalada)
    if not exito:
        raise HTTPException(status_code=404, detail="No se han procesado imagenes DNI")
    img_bytes = buffer.tobytes()

    return StreamingResponse(
        io.BytesIO(img_bytes),
        media_type="image/webp"
    )

# --- Endpoint de imagen: sirve cada imagen por separado ---
@app.get("/imagen/selfie/")
def get_imagen_sefie() -> StreamingResponse:
    fotoVerify.createCoparacionImg()

    #imagenSelfie_reescalada = cv2.resize(fotoVerify.coparacionImg, (640, 300), interpolation=cv2.INTER_AREA)
    imagenSelfie_reescalada = fotoVerify.coparacionImg
    exito, buffer = cv2.imencode('.webp', imagenSelfie_reescalada)
    if not exito:
        raise HTTPException(status_code=404, detail="No se han procesado imagenes Selfie")
    img_bytes = buffer.tobytes()

    return StreamingResponse(
        io.BytesIO(img_bytes),
        media_type="image/webp"
    )

@app.post("/spoofing/")
async def procesar_imagenes( imgSlf: UploadFile = File(...) ):
    imgSlf64 = await imgSlf.read()
    imgSlfNpi = np.frombuffer(imgSlf64, np.uint8)
    imgSlfCv2 = cv2.imdecode(imgSlfNpi, cv2.IMREAD_COLOR)

    if imgSlfCv2 is None:
        return ResultadoDNI(
            status = False,
            code = "El archivo enviado no es una imagen válida o está corrupto.",
            clases = []
        )

    ok, frame = fotoVerify.verirySpoofing(imgSlfCv2)
    if not ok:
        return ResultadoDNI(
            status = False,
            code = "No es un Selfie.",
            clases = []
        )

    if frame is None:
        raise HTTPException(status_code=404, detail="No es un Selfie")

    nuevo_alto = 300
    alto, ancho = frame.shape[:2]
    nuevo_ancho = int((nuevo_alto / alto) * ancho)
    img_resize = cv2.resize(frame, (nuevo_ancho, nuevo_alto))
    exito, buffer = cv2.imencode('.webp', img_resize)
    if not exito:
        raise HTTPException(status_code=404, detail="No se han procesado Selfie")
    img_bytes = buffer.tobytes()
    return StreamingResponse(
        io.BytesIO(img_bytes),
        media_type="image/webp"
    )
