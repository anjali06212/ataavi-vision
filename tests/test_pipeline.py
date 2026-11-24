import pytest
import numpy as np
from PIL import Image
from src.augmentations import RealWorldDegradations
from src.detector import BirdDetector
from src.classifier import BirdClassifier, Prediction, ClassificationResult
from src.pipeline import AtaaviVisionPipeline, PipelineOutput

def create_synthetic_image(width=400, height=300, color=(120, 150, 100)) -> Image.Image:
    """Create a test image in RGB mode."""
    arr = np.full((height, width, 3), color, dtype=np.uint8)
    return Image.fromarray(arr)

def test_augmentations_shape_and_mode():
    img = create_synthetic_image(200, 150)
    
    # 1. low_light
    deg_ll = RealWorldDegradations.low_light(img)
    assert deg_ll.size == img.size
    assert deg_ll.mode == "RGB"
    
    # 2. motion_blur
    deg_mb = RealWorldDegradations.motion_blur(img)
    assert deg_mb.size == img.size
    assert deg_mb.mode == "RGB"
    
    # 3. foliage_occlusion
    deg_fo = RealWorldDegradations.foliage_occlusion(img)
    assert deg_fo.size == img.size
    assert deg_fo.mode == "RGB"
    
    # 4. distant_bird
    deg_db = RealWorldDegradations.distant_bird(img)
    assert deg_db.size == img.size
    assert deg_db.mode == "RGB"

def test_detector_padding_and_fallback():
    detector = BirdDetector(model_name="yolov8n.pt", conf_thresh=0.99) # high threshold ensures fallback on plain color
    img = create_synthetic_image(300, 200)
    
    res = detector.detect(img)
    # Check fallback behavior
    assert res.fallback_used is True
    assert len(res.crops) == 1
    assert res.crops[0].size == img.size
    assert res.boxes[0] == (0, 0, 300, 200)

def test_classifier_output_contract():
    classifier = BirdClassifier(species_json_path="data/indian_birds.json")
    img = create_synthetic_image(224, 224)
    
    results = classifier.classify([img], top_k=3)
    assert len(results) == 1
    res = results[0]
    assert len(res.top_predictions) == 3
    assert isinstance(res.top_predictions[0], Prediction)
    assert 0.0 <= res.top_predictions[0].confidence <= 1.0
    assert len(res.embedding) > 0
    # L2 norm check
    norm = np.linalg.norm(res.embedding)
    assert np.isclose(norm, 1.0, atol=1e-3)

def test_pipeline_end_to_end():
    pipeline = AtaaviVisionPipeline(species_json_path="data/indian_birds.json")
    img = create_synthetic_image(320, 240)
    
    out: PipelineOutput = pipeline.predict(img)
    assert isinstance(out, PipelineOutput)
    assert out.annotated_image.size == img.size
    assert len(out.detections) >= 1
    assert out.total_ms > 0
    assert "top_predictions" in out.detections[0]
