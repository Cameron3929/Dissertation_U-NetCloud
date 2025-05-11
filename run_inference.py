import os, numpy as np
from PIL import Image

import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

# Logger & runtime
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
runtime    = trt.Runtime(TRT_LOGGER)

# load your engine
engine_path = os.path.expanduser('~/Documents/unet95_final.trt')
with open(engine_path, 'rb') as f:
    engine = runtime.deserialize_cuda_engine(f.read())

# allocate I/O buffers
context = engine.create_execution_context()
# assume 1 input, 1 output
for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding))
    dtype = trt.nptype(engine.get_binding_dtype(binding))
    # allocate host and device buffers
    if engine.binding_is_input(binding):
        h_input = np.empty(size, dtype=dtype)
        d_input = cuda.mem_alloc(h_input.nbytes)
    else:
        h_output = np.empty(size, dtype=dtype)
        d_output = cuda.mem_alloc(h_output.nbytes)

# path to your patches folders
base_dir = os.path.expanduser('~/Documents/inference_patches')
scenes   = [os.path.join(base_dir,d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir,d))]

for scene in scenes:
    # load and stack BGRN into [1,C,H,W]
    bands = {}
    for fn in os.listdir(scene):
        if 'blue' in fn: bands['blue']=fn
        if 'green'in fn: bands['green']=fn
        if 'red'  in fn: bands['red']=fn
        if 'nir'  in fn: bands['nir']=fn

    img = np.stack([
        np.array(Image.open(os.path.join(scene,bands[k]))) for k in ('blue','green','red','nir')
    ], axis=0).astype(np.float32)/10000.0
    img = np.expand_dims(img, axis=0)

    # copy input to device, execute, copy output back
    cuda.memcpy_htod(d_input, img.ravel())
    context.execute_v2([int(d_input), int(d_output)])
    cuda.memcpy_dtoh(h_output, d_output)

    # threshold & save mask
    mask = (h_output.reshape(engine.get_binding_shape(engine.get_binding_index(binding))))>0.55
    out_path = os.path.join(scene,'predicted_mask.png')
    Image.fromarray((mask*255).astype(np.uint8)).save(out_path)
    print(f'✅ Saved {out_path}')
