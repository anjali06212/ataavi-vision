import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
from PIL import Image
import numpy as np
from ultralytics import YOLO

COCO_BIRD_CLASS_ID = 14

@dataclass
class DetectionResult:
    crops: List[Image.Image]
    boxes: List[Tuple[int, int, int, int]]  # [x1, y1, x2, y2]
    confidences: List[float]
    fallback_used: bool
    latency_ms: float

class BirdDetector:
    """
    Ultralytics YOLOv8 detector constrained to avian detection (COCO class 14).
    Applies configurable confidence thresholding, boundary-clamped margin padding,
    and automatic full-image fallback if no bird is detected.
    """
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_thresh: float = 0.20,
        pad_pct: float = 0.15,
        device: Optional[str] = None
    ):
        self.conf_thresh = conf_thresh
        self.pad_pct = pad_pct
        self.device = device
        # Initializes YOLO model (auto-downloads weights if not present)
        self.model = YOLO(model_name)

    def detect(
        self,
        image: Image.Image,
        conf_thresh: Optional[float] = None,
        pad_pct: Optional[float] = None
    ) -> DetectionResult:
        t0 = time.perf_counter()
        thresh = conf_thresh if conf_thresh is not None else self.conf_thresh
        pad = pad_pct if pad_pct is not None else self.pad_pct

        img_w, img_h = image.size
        
        # Run YOLO inference
        results = self.model.predict(
            source=image,
            conf=thresh,
            classes=[COCO_BIRD_CLASS_ID],
            verbose=False,
            device=self.device
        )

        crops: List[Image.Image] = []
        boxes: List[Tuple[int, int, int, int]] = []
        confidences: List[float] = []

        if results and len(results[0].boxes) > 0:
            res_boxes = results[0].boxes
            for b in res_boxes:
                xyxy = b.xyxy[0].cpu().numpy()
                conf = float(b.conf[0].cpu().numpy())

                x1, y1, x2, y2 = xyxy
                box_w = x2 - x1
                box_h = y2 - y1

                # Apply margin padding
                pad_x = box_w * pad
                pad_y = box_h * pad

                padded_x1 = max(0, int(round(x1 - pad_x)))
                padded_y1 = max(0, int(round(y1 - pad_y)))
                padded_x2 = min(img_w, int(round(x2 + pad_x)))
                padded_y2 = min(img_h, int(round(y2 + pad_y)))

                crop = image.crop((padded_x1, padded_y1, padded_x2, padded_y2))
                crops.append(crop)
                boxes.append((padded_x1, padded_y1, padded_x2, padded_y2))
                confidences.append(conf)

        fallback_used = False
        if len(crops) == 0:
            # Fallback: No bird detected with sufficient confidence, fall back to entire image
            fallback_used = True
            crops.append(image)
            boxes.append((0, 0, img_w, img_h))
            confidences.append(0.0)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return DetectionResult(
            crops=crops,
            boxes=boxes,
            confidences=confidences,
            fallback_used=fallback_used,
            latency_ms=latency_ms
        )
