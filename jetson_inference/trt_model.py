from loguru import logger as LOGGER
import tensorrt as trt
import argparse
import os
from pathlib import Path
from collections import namedtuple
from loguru import logger as LOGGER
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import numpy as np

def file_size(path):
    # Return file/dir size (MB)
    mb = 1 << 20  # bytes to MiB (1024 ** 2)
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / mb
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / mb
    else:
        return 0.0

def rename(path, new_name):
    suffix = path.suffix
    return path.with_name(f'{new_name}.t').with_suffix(suffix)

def make_parser():
    parser = argparse.ArgumentParser("Configuration")
    parser.add_argument("--model_path", required=True, type=str, help="model path")
    parser.add_argument("--workspace",default=1<<28, type=int, help="workspace")
    return parser

class TrtModel():
    def __init__(self,engine_path,cuda_index=0,channel_first=True,**kwargs):
        number_of_gpus = cuda.Device.count()

        if type(cuda_index) == int:
            self.cuda_ctx = cuda.Device(cuda_index).make_context()
        else:
            self.cuda_ctx = cuda_index

        self.stream = cuda.Stream()

        self.channel_first = channel_first

        logger = trt.Logger(trt.Logger.WARNING)
        trt_runtime = trt.Runtime(logger)


        #Load model
        trt.init_libnvinfer_plugins(logger, "")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()

        LOGGER.info(f"SELECT GPU: {self.cuda_ctx.get_device().MULTI_GPU_BOARD_GROUP_ID} over GPUs {list(range(number_of_gpus))}")
        self.engine = trt_runtime.deserialize_cuda_engine(engine_data)

        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

        LOGGER.info("Acquired %.5f GB"%(self.engine.device_memory_size/1e9))

        overlapped_attributes = {'engine','inputs','outputs','bindings','stream','context'}.intersection(set(kwargs.keys()))
        assert not overlapped_attributes, f"Can't overlap attributes: {overlapped_attributes}"
        self.__dict__.update((k, v) for k, v in kwargs.items())
        self.cuda_ctx.pop()

    def allocate_buffers(self):
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'host_mem', 'device_mem'))
        inputs = []
        outputs = []
        bindings = []

        stream = self.stream

        for index in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(index)
            dtype = trt.nptype(self.engine.get_binding_dtype(index))
            shape = tuple(self.engine.get_binding_shape(index))
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            if self.engine.binding_is_input(self.engine[index]):
                inputs.append(Binding(name, dtype, shape, host_mem, device_mem))
            else:
                outputs.append(Binding(name, dtype, shape, host_mem, device_mem))
        LOGGER.debug(f'Inputs {inputs}')
        LOGGER.debug(f'Inputs {outputs}')

        return inputs, outputs, bindings, stream

    # @time_ticker
    def __call__(self,im,batch_size=1):
        self.cuda_ctx.push()
        assert im.shape == self.inputs[0].shape, (im.shape, self.inputs[0].shape)

#         self.inputs['images'] = int(im.data_ptr())
#         self.context.execute_v2(list(self.binding_addrs.values()))
#         y = self.bindings['output'].data


        im = im.astype(self.inputs[0].dtype)

        np.copyto(self.inputs[0].host_mem,im.ravel())
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device_mem, inp.host_mem, self.stream)

        #execute_async_v2 ignore the batch_size
        self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host_mem, out.device_mem, self.stream)

        self.stream.synchronize()
        self.cuda_ctx.pop()
        output = {out.name: out.host_mem.reshape(out.shape) for out in self.outputs}


        return output

    def preprocess(self,image,shape,mean=[],stddev=[],zero_pad=[]):
        data = image.copy()
        # Zero Pad
        if zero_pad:
            pad_0 = [0,0,0]
            data = cv2.copyMakeBorder(data,zero_pad[0],zero_pad[1],zero_pad[2],zero_pad[3],cv2.BORDER_CONSTANT,value=pad_0)

        data = cv2.resize(data,shape)
        if mean and stddev:
            # Mean normalization
            mean = np.array(mean).astype('float32')
            stddev = np.array(stddev).astype('float32')
            data = (np.asarray(data).astype('float32') - mean) / stddev
        return data

    def postprocess(self,output,**kwargs):
        return
