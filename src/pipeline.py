import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from src.detector import BirdDetector, DetectionResult
from src.classifier import BirdClassifier, ClassificationResult

@dataclass
class PipelineOutput:
    annotated_image: Image.Image
    detections: List[Dict[str, Any]]
    det_ms: float
    cls_ms: float
    total_ms: float
    fallback_used: bool

class AtaaviVisionPipeline:
    """
    Two-stage inference pipeline:
    Stage 1: YOLOv8 Avian Detector (localizes birds with padded bounding boxes).
    Stage 2: BioCLIP Zero-Shot Classifier (identifies Indian bird species).
    """
    def __init__(
        self,
        detector: Optional[BirdDetector] = None,
        classifier: Optional[BirdClassifier] = None,
        species_json_path: str = "data/indian_birds.json"
    ):
        self.detector = detector if detector is not None else BirdDetector()
        self.classifier = classifier if classifier is not None else BirdClassifier(species_json_path=species_json_path)

    def predict(
        self,
        image: Image.Image,
        conf_thresh: Optional[float] = None,
        pad_pct: Optional[float] = None,
        top_k: int = 3
    ) -> PipelineOutput:
        t_start = time.perf_counter()

        # Ensure image is in RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Stage 1: Detection
        det_res: DetectionResult = self.detector.detect(
            image,
            conf_thresh=conf_thresh,
            pad_pct=pad_pct
        )

        # Stage 2: Zero-shot Classification on crops
        cls_results: List[ClassificationResult] = self.classifier.classify(
            det_res.crops,
            top_k=top_k
        )

        cls_ms = sum(r.latency_ms for r in cls_results)

        # Format detection objects
        detections = []
        for i, (box, crop, det_conf, cls_res) in enumerate(zip(
            det_res.boxes, det_res.crops, det_res.confidences, cls_results
        )):
            detections.append({
                "box": box,
                "crop": crop,
                "detection_confidence": det_conf,
                "top_predictions": cls_res.top_predictions,
                "embedding": cls_res.embedding
            })

        # Draw annotations on visualization copy
        annotated_img = image.copy()
        draw = ImageDraw.Draw(annotated_img)

        # Try to load a clean default font
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            top_pred = det["top_predictions"][0]
            label = f"{top_pred.species} ({top_pred.confidence*100:.1f}%)"

            if not det_res.fallback_used:
                # Green bounding box for detected birds
                draw.rectangle([x1, y1, x2, y2], outline=(0, 230, 115), width=3)
                # Label badge background
                text_bbox = draw.textbbox((x1, max(0, y1 - 18)), label, font=font)
                draw.rectangle(
                    [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2],
                    fill=(0, 230, 115)
                )
                draw.text((x1, max(0, y1 - 18)), label, fill=(10, 10, 10), font=font)
            else:
                # Orange badge showing full-image fallback
                fallback_label = f"[Full Image Fallback] {label}"
                draw.rectangle([x1, y1, x2, y2], outline=(255, 140, 0), width=2)
                draw.text((10, 10), fallback_label, fill=(255, 140, 0), font=font)

        total_ms = (time.perf_counter() - t_start) * 1000.0

        return PipelineOutput(
            annotated_image=annotated_img,
            detections=detections,
            det_ms=det_res.latency_ms,
            cls_ms=cls_ms,
            total_ms=total_ms,
            fallback_used=det_res.fallback_used
        )
