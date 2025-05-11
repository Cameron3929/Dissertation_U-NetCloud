# common.py
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def load_engine(file_path):
    with open(file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    """Allocate host/device buffers for all bindings."""
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for idx in range(engine.num_bindings):
        name = engine.get_binding_name(idx)
        shape = engine.get_binding_shape(idx)
        dtype = trt.nptype(engine.get_binding_dtype(idx))
        # total # of elements in buffer
        size = int(np.prod(shape))
        # host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        dev_mem   = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(dev_mem))
        if engine.binding_is_input(idx):
            inputs.append({"host": host_mem, "device": dev_mem})
        else:
            outputs.append({"host": host_mem, "device": dev_mem})
    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream):
    # copy input → GPU
    for buf in inputs:
        cuda.memcpy_htod_async(buf["device"], buf["host"], stream)
    # run
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # copy GPU → output host
    for buf in outputs:
        cuda.memcpy_dtoh_async(buf["host"], buf["device"], stream)
    stream.synchronize()
    # return only the host outputs
    return [out["host"] for out in outputs]
