import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np
import torch
import open_clip

@dataclass
class Prediction:
    species: str
    scientific_name: str
    confidence: float
    order: str = ""
    family: str = ""

@dataclass
class ClassificationResult:
    top_predictions: List[Prediction]
    embedding: np.ndarray
    latency_ms: float

class BirdClassifier:
    """
    BioCLIP Zero-Shot Taxonomic Classifier constrained to Indian avifauna.
    Uses Imageomics BioCLIP (hf-hub:imageomics/bioclip) through OpenCLIP with
    precomputed text embeddings for high-throughput, low-latency evaluation.
    """
    def __init__(
        self,
        species_json_path: str = "data/indian_birds.json",
        model_name: str = "hf-hub:imageomics/bioclip",
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Indian bird metadata
        with open(species_json_path, "r", encoding="utf-8") as f:
            self.species_catalog = json.load(f)
            
        print(f"Loading BioCLIP model '{model_name}' on {self.device}...")
        try:
            self.model, self.preprocess_train, self.preprocess = open_clip.create_model_and_transforms(
                model_name,
                device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
        except Exception as e:
            # Fallback for standard ViT-B-16 if network download is blocked or fails
            print(f"Warning: Failed to load {model_name} ({e}). Falling back to ViT-B-16 / openai...")
            self.model, self.preprocess_train, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-B-16",
                pretrained="openai",
                device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer("ViT-B-16")

        self.model.eval()
        self._encode_species_prompts()

    def _encode_species_prompts(self):
        """Precompute normalized text embeddings for all target species."""
        # Using domain-aligned taxonomic zero-shot prompt templates
        prompts = [
            f"a photo of a {bird['common_name']} ({bird['scientific_name']}), a bird species in family {bird.get('family', 'Aves')}"
            for bird in self.species_catalog
        ]
        
        text_tokens = self.tokenizer(prompts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        self.text_features = text_features # Shape: (N_species, D)

    def classify(self, crops: List[Image.Image], top_k: int = 3) -> List[ClassificationResult]:
        """Classify a batch of cropped bird images against the Indian species catalog."""
        if not crops:
            return []

        t0 = time.perf_counter()
        
        # Preprocess images
        tensor_batch = torch.stack([self.preprocess(crop) for crop in crops]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(tensor_batch)
            # L2-normalize
            image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)

            # Cosine similarity matrix scaled by model temperature
            similarity = (100.0 * image_features_norm @ self.text_features.T).softmax(dim=-1)

        sim_np = similarity.cpu().numpy()
        emb_np = image_features_norm.cpu().numpy()
        total_time_ms = (time.perf_counter() - t0) * 1000.0
        time_per_crop = total_time_ms / len(crops)

        results = []
        for i in range(len(crops)):
            probs = sim_np[i]
            top_indices = np.argsort(probs)[::-1][:top_k]

            preds = []
            for idx in top_indices:
                species_info = self.species_catalog[idx]
                preds.append(
                    Prediction(
                        species=species_info["common_name"],
                        scientific_name=species_info["scientific_name"],
                        confidence=float(probs[idx]),
                        order=species_info.get("order", ""),
                        family=species_info.get("family", "")
                    )
                )

            results.append(
                ClassificationResult(
                    top_predictions=preds,
                    embedding=emb_np[i],
                    latency_ms=time_per_crop
                )
            )

        return results

    def get_embeddings(self, crops: List[Image.Image]) -> np.ndarray:
        """Return L2-normalized image embeddings for feature analysis."""
        if not crops:
            return np.empty((0, 512), dtype=np.float32)

        tensor_batch = torch.stack([self.preprocess(crop) for crop in crops]).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(tensor_batch)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()
