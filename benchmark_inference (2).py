
import os
import time
import numpy as np
from PIL import Image
import subprocess
import csv
import psutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_inference_trt import load_engine, allocate_buffers, do_inference
import tifffile as tiff

def read_power():
    try:
        with open("/sys/bus/i2c/drivers/ina3221x/6-0040/iio_device/in_power0_input") as f:
            return int(f.read().strip()) / 1_000_000
    except:
        return -1

def read_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000
    except:
        return -1

def read_memory():
    return psutil.virtual_memory().used / (1024 ** 2)

def dice_score(pred, truth):
    pred = pred > 127
    truth = truth > 127
    intersection = np.logical_and(pred, truth).sum()
    return 2.0 * intersection / (pred.sum() + truth.sum() + 1e-6)

def iou_score(pred, truth):
    pred = pred > 127
    truth = truth > 127
    intersection = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()
    return intersection / (union + 1e-6)

INPUT_DIR = "Patches"
PREDICTION_DIR = "inference_results"
os.makedirs(PREDICTION_DIR, exist_ok=True)
OUTPUT_CSV = "benchmark_results.csv"
MODEL_PATH = "unet95_final.trt"

engine = load_engine(MODEL_PATH)
context = engine.create_execution_context()
inputs, outputs, bindings, stream = allocate_buffers(engine)

results = [("filename", "latency_ms", "temp_before_C", "temp_after_C", "power_before_W", "power_after_W", "mem_used_MB", "dice", "iou")]

latencies, temps, powers, mems, dices, ious = [], [], [], [], [], []

for scene in sorted(os.listdir(INPUT_DIR)):
    scene_dir = os.path.join(INPUT_DIR, scene)
    if not os.path.isdir(scene_dir):
        continue

    print(f"[DEBUG] Checking {scene_dir}")
    try:
        bands = {}
        for fn in os.listdir(scene_dir):
            fnl = fn.lower()
            if fnl.endswith(".tif"):
                for k in ("blue", "green", "red", "nir"):
                    if k in fnl:
                        bands[k] = fn
        if set(bands) != {"blue", "green", "red", "nir"}:
            print(f"[SKIP] Incomplete bands in {scene}")
            continue

        arrs = []
        for k in ("blue", "green", "red", "nir"):
            path = os.path.join(scene_dir, bands[k])
            try:
                img = tiff.imread(path)
            except Exception as e:
                print(f"[ERROR] Could not read {path} with tifffile: {e}")
                raise e
            arrs.append(img)

        x = np.stack(arrs, axis=0).astype(np.float32) / 10000.0
        x = np.expand_dims(x, axis=0)

        temp_before = read_temp()
        power_before = read_power()
        mem_before = read_memory()

        start = time.time()
        inputs[0]["host"] = x
        y = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)[0]
        latency_ms = (time.time() - start) * 1000

        temp_after = read_temp()
        power_after = read_power()
        mem_after = read_memory()

        mem_used = mem_after - mem_before

        y = y.reshape((1, 1, x.shape[2], x.shape[3]))
        mask = (y[0, 0] > 0.5).astype(np.uint8) * 255

        pred_path = os.path.join(PREDICTION_DIR, f"{scene}_predicted.png")
        Image.fromarray(mask).save(pred_path)

        gt_candidates = [f for f in os.listdir(scene_dir) if f.lower().startswith("gt_patch") and f.lower().endswith(".tif")]
        if gt_candidates:
            gt_path = os.path.join(scene_dir, gt_candidates[0])
            try:
                truth = np.array(Image.open(gt_path).resize(mask.shape[::-1]))
                dice = dice_score(mask, truth)
                iou = iou_score(mask, truth)
            except Exception as e:
                print(f"[ERROR] Failed to process GT mask {gt_path}: {e}")
                dice, iou = None, None
        else:
            dice, iou = None, None
            print(f"[WARNING] No ground truth found for {scene}")

        results.append((scene, round(latency_ms, 2), temp_before, temp_after, power_before, power_after, round(mem_used, 2), dice, iou))

        latencies.append(latency_ms)
        temps.append(temp_after)
        powers.append(power_after)
        mems.append(mem_used)
        if dice is not None: dices.append(dice)
        if iou is not None: ious.append(iou)

        print(f"[OK] {scene} processed")

    except Exception as e:
        print(f"[ERROR] {scene_dir}: {e}")

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(results)

print("[SUMMARY]")
print("latencies:", latencies)
print("temps:", temps)
print("powers:", powers)
print("mems:", mems)
print("dices:", dices)
print("ious:", ious)

latencies = [l for l in latencies if l is not None]
temps = [t for t in temps if t is not None]
powers = [p for p in powers if p is not None]
mems = [m for m in mems if m is not None]

plt.figure()
if latencies:
    plt.plot(latencies, label="Latency (ms)")
if temps:
    plt.plot(temps, label="Temp (°C)")
if powers:
    plt.plot(powers, label="Power (W)")
if mems:
    plt.plot(mems, label="Memory Used (MB)")
if dices:
    plt.plot(dices, label="Dice")
if ious:
    plt.plot(ious, label="IoU")

plt.xlabel("Image Index")
plt.legend()
plt.title("Jetson Nano Inference Benchmarking")
plt.savefig("benchmark_summary_plot.png")

print(f"Benchmarking complete. Results saved to {OUTPUT_CSV} and 'benchmark_summary_plot.png'")
