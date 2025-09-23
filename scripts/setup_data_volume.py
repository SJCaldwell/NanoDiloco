import os
import json
import time
import modal

app = modal.App("hf-c4-tiny-volume")

VOLUME_NAME = "hf-c4-tiny"
vol = modal.Volume.from_name(VOLUME_NAME)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "datasets",
        "huggingface-hub",
        "pyarrow",
        "tqdm",
        "xxhash",
        "zstandard"
    )
)

VOL = "/vol"
SAVE_DIR = f"{VOL}/datasets/PrimeIntellect/c4-tiny/en/save_to_disk"
CACHE_DIR = f"{VOL}/hf_cache"

@app.function(image=image, volumes={VOL: vol}, timeout=3600)
def materialize_c4_tiny():
    # Put all HF caches inside the volume so later jobs can use them offline
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
    os.environ["HF_HOME"] = f"{VOL}/hf_home"
    os.environ["HF_HUB_CACHE"] = f"{os.environ['HF_HOME']}/hub"

    from datasets import load_dataset

    ds = load_dataset(
        path="PrimeIntellect/c4-tiny",
        name="en",
        cache_dir=CACHE_DIR,
        verification_mode="no_checks"
    )

    ds.save_to_disk(SAVE_DIR)

    # Human-friendly manifest
    meta = {
        "source": "PrimeIntellect/c4-tiny",
        "config": "en",
        "paths": {"save_to_disk": SAVE_DIR, "hf_cache": CACHE_DIR},
        "created_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    }
    manifest_path = f"{VOL}/datasets/PrimeIntellect/c4-tiny/manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(meta, f, indent=2)

@app.local_entrypoint()
def main():
    materialize_c4_tiny.remote()
    print(f"Volume '{VOLUME_NAME}' now contains c4-tiny at {SAVE_DIR}")