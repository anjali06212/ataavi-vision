import os
import sys
import time
import json
import csv
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from PIL import Image
import io

DATA_DIR = Path(__file__).parent.parent / "data"
BIRDS_JSON_PATH = DATA_DIR / "indian_birds.json"
TEST_IMAGES_DIR = DATA_DIR / "test_images"
METADATA_CSV_PATH = DATA_DIR / "metadata.csv"

TARGET_SPECIES_INDICES = [
    0,   # Indian Roller
    1,   # Malabar Grey Hornbill
    2,   # White-throated Kingfisher
    3,   # Indian Pitta
    4,   # Rufous Treepie
    5,   # Asian Paradise-Flycatcher
    6,   # Greater Coucal
    7,   # Oriental Magpie-Robin
    8,   # Black Drongo
    9,   # Red-whiskered Bulbul
    10,  # Purple Sunbird
    11,  # Green Bee-eater
    12,  # Painted Stork
    13,  # Brahminy Kite
    14,  # Spotted Owlet
]

IMAGES_PER_SPECIES = 8
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=2)
session.mount("https://", adapter)
session.headers.update({"User-Agent": USER_AGENT})

def fetch_obs(sci_name: str) -> List[Dict[str, Any]]:
    url = "https://api.inaturalist.org/v1/observations"
    params = {
        "taxon_name": sci_name,
        "quality_grade": "research",
        "photos": "true",
        "per_page": 30,
        "order": "desc",
        "order_by": "votes"
    }
    for _ in range(3):
        try:
            r = session.get(url, params=params, timeout=8)
            if r.status_code == 200:
                return r.json().get("results", [])
        except Exception:
            time.sleep(0.5)
    return []

def download_img(url: str):
    try:
        r = session.get(url, timeout=8)
        if r.status_code == 200:
            b = r.content
            img = Image.open(io.BytesIO(b))
            img.verify()
            return Image.open(io.BytesIO(b)).convert("RGB")
    except Exception:
        pass
    return None

def process_species(bird_info):
    common_name = bird_info["common_name"]
    sci_name = bird_info["scientific_name"]
    slug = common_name.lower().replace(" ", "_").replace("-", "_")
    species_dir = TEST_IMAGES_DIR / slug
    species_dir.mkdir(parents=True, exist_ok=True)

    existing = list(species_dir.glob("*.jpg"))
    if len(existing) >= IMAGES_PER_SPECIES:
        print(f"[Done] {common_name} ({len(existing)} imgs)")
        rows = []
        for i, f in enumerate(sorted(existing)):
            rows.append({
                "species_slug": slug,
                "common_name": common_name,
                "scientific_name": sci_name,
                "filename": f.name,
                "relative_path": f"data/test_images/{slug}/{f.name}",
                "width": 640,
                "height": 480,
                "source_url": "iNaturalist",
                "attribution": "iNaturalist contributor"
            })
        return rows

    needed = IMAGES_PER_SPECIES - len(existing)
    print(f"[Fetching] {common_name}: need {needed} more...")
    obs_list = fetch_obs(sci_name)
    saved = len(existing)
    rows = []

    # Keep track of existing filenames
    for f in existing:
        rows.append({
            "species_slug": slug,
            "common_name": common_name,
            "scientific_name": sci_name,
            "filename": f.name,
            "relative_path": f"data/test_images/{slug}/{f.name}",
            "width": 640,
            "height": 480,
            "source_url": "iNaturalist",
            "attribution": "iNaturalist contributor"
        })

    for obs in obs_list:
        if saved >= IMAGES_PER_SPECIES:
            break
        for p in obs.get("photos", []):
            if saved >= IMAGES_PER_SPECIES:
                break
            url = p.get("url")
            if not url:
                continue
            med_url = url.replace("square.", "medium.").replace("small.", "medium.")
            img = download_img(med_url)
            if img is None:
                continue

            fname = f"{slug}_{saved + 1:02d}.jpg"
            fpath = species_dir / fname
            img.save(fpath, "JPEG", quality=90)
            rows.append({
                "species_slug": slug,
                "common_name": common_name,
                "scientific_name": sci_name,
                "filename": fname,
                "relative_path": f"data/test_images/{slug}/{fname}",
                "width": img.width,
                "height": img.height,
                "source_url": med_url,
                "attribution": p.get("attribution", "iNaturalist contributor")
            })
            saved += 1

    print(f"[Finished] {common_name}: {saved}/{IMAGES_PER_SPECIES} images")
    return rows

def main():
    with open(BIRDS_JSON_PATH, "r", encoding="utf-8") as f:
        all_birds = json.load(f)
    target_birds = [all_birds[i] for i in TARGET_SPECIES_INDICES]

    all_rows = []
    # Concurrently fetch across species
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_species, b) for b in target_birds]
        for fut in as_completed(futures):
            res = fut.result()
            all_rows.extend(res)

    unique_rows = {r["relative_path"]: r for r in all_rows}.values()
    with open(METADATA_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["species_slug", "common_name", "scientific_name", "filename", "relative_path", "width", "height", "source_url", "attribution"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique_rows)

    print(f"\nAll done! Successfully registered {len(unique_rows)} images in {METADATA_CSV_PATH}")

if __name__ == "__main__":
    main()
