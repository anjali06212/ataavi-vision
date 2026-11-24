FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download models during build
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); import open_clip; open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')"

EXPOSE 7860
CMD ["python", "app.py"]
