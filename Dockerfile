FROM ultralytics/ultralytics:latest

RUN mkdir -p /root/.deepface/weights/
COPY pesos/facenet512_weights.h5 /root/.deepface/weights/facenet512_weights.h5

WORKDIR /app

COPY requirements.txt .

RUN python3 -m pip install --root-user-action=ignore --upgrade pip
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# Copia todos los archivos que terminen en .pt a la carpeta actual (/app)
COPY pesos/*.pt ./
COPY *.py ./
COPY *.jpg ./

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
