#!/usr/bin/env python3
# run_inference_trt.py

import os
import numpy as np
import tifffile as tiff
from PIL import Image  # <-- NEW import

from common import load_engine, allocate_buffers, do_inference

# --- CONFIG ---
MODEL_PATH = "unet95_final.trt"
BASE_DIR = "inference_patches"
OUTPUT_DIR = "inference_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load your TRT engine & create context + buffers ---
engine = load_engine(MODEL_PATH)
context = engine.create_execution_context()
inputs, outputs, bindings, stream = allocate_buffers(engine)

# --- Process each sub-folder ---
for scene in sorted(os.listdir(BASE_DIR)):
    scene_dir = os.path.join(BASE_DIR, scene)
    if not os.path.isdir(scene_dir):
        continue

    try:
        # 1) Find exactly four TIFFs named 'blue','green','red','nir'
        bands = {}
        for fn in os.listdir(scene_dir):
            fnl = fn.lower()
            if fnl.endswith((".tif", ".tiff")):
                for k in ("blue", "green", "red", "nir"):
                    if k in fnl:
                        bands[k] = fn
        assert set(bands) == {"blue", "green", "red", "nir"}, \
            f"Expected exactly 4 bands in {scene_dir}, got {bands.keys()}"

        # 2) Read & stack -> (C, H, W)
        arrs = []
        for k in ("blue", "green", "red", "nir"):
            img = tiff.imread(os.path.join(scene_dir, bands[k]))
            arrs.append(img)
        x = np.stack(arrs, axis=0).astype(np.float32)

        # 3) Normalize input if needed (e.g., to 0-1)
        x = x / 10000.0  # <- adjust depending on how model was trained

        # 4) Add batch dimension (N, C, H, W)
        x = np.expand_dims(x, axis=0)

        # 5) Run inference
        inputs[0]['host'] = x
        y = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)[0]

        # 6) Post-process output
        y = y.reshape((1, 1, x.shape[2], x.shape[3]))  # (N, 1, H, W)
        mask = y[0, 0]  # (H, W)

        # 7) Threshold mask
        mask = (mask > 0.5).astype(np.uint8) * 255  # binary 0 or 255

        # 8) Save mask PROPERLY as a PNG
        out_img = Image.fromarray(mask)
        output_path = os.path.join(OUTPUT_DIR, f"{scene}_mask.png")
        out_img.save(output_path)

        print(f"[OK] {output_path}")

    except Exception as e:
        print(f"[ERROR] {scene_dir}: {e}")

