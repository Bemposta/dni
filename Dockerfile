FROM ultralytics/ultralytics:latest

RUN apt-get update && apt-get install -y wget

RUN mkdir -p /root/.deepface/weights/
COPY pesos/*.h5 /root/.deepface/weights/
COPY pesos/*.pth /root/.deepface/weights/
RUN wget -q -P /root/.deepface/weights/ https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5

WORKDIR /app

COPY requirements.txt .

RUN python3 -m pip install --root-user-action=ignore --upgrade pip
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# Copia todos los archivos que terminen en .pt a la carpeta actual (/app)
COPY *.jpg ./
COPY pesos/*.pt ./
COPY *.py ./
COPY *.html ./

CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]


