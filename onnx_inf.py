from abc import ABC, abstractmethod

from functools import partial
import cv2
import onnx
import onnxruntime as ort
import numpy as np
from tqdm import tqdm
import os
# import matplotlib.pyplot as plt
# import matplotlib as mpl

# mpl.rcParams['figure.figsize'] = (12,12)
# mpl.rcParams['axes.grid'] = False


class Skechlize(ABC):
    def __init__(self, gpu_id=-1):
        self.gpu_id = gpu_id
        return
    
    def __call__(self, src_data_dict):
        out_put_data = src_data_dict.copy()
        self.preprocess(out_put_data)
        self.inference(out_put_data)
        self.postprocess(out_put_data)
        return out_put_data

    @abstractmethod
    def preprocess(self, data_dict):
        return 
    
    @abstractmethod
    def postprocess(self, data_dict):
        return
    
    @abstractmethod
    def inference(self, data_dict):
        return
    

class OnnxSimpleGrayProcess(Skechlize):
    def __init__(
        self,
        onnx_model_fn,
        wh=(768,768),
        mean=127.5,
        std=127.5,
        gpu_id=-1):
        super().__init__(gpu_id)
        self.inf_wh = wh
        self.normalize = lambda x: (x.astype(np.float32)- mean)/ std
        self.denormlize = lambda x:(np.clip(x * std + mean,0,255)).astype(np.uint8)
        self.onnx_model = onnx.load_model(onnx_model_fn)

        if(self.gpu_id >= 0):
            _provider = ['CUDAExecutionProvider']
        else:
            _provider = ['CPUExecutionProvider']
        self.sess = ort.InferenceSession(self.onnx_model.SerializeToString(),providers=_provider)

        self.input_names = [x.name for x in self.sess.get_inputs()]
        self.output_names = [x.name for x in self.sess.get_outputs()]

        print(f'model: {onnx_model_fn} loaded.\n\tinput nodes are: {self.input_names},\n\toutput nodes are {self.output_names}')

        return
    
    def rsz_padding(self, srcimg):
        src_h, src_w = srcimg.shape[:2]
    
        r = min(float(self.inf_wh[1]) / float(src_h),
                float(self.inf_wh[0]) / float(src_w))

        dst_w = int(src_w * r)
        dst_h = int(src_h * r)
        resized_img = cv2.resize(
            srcimg, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
        
        pad_l = int((self.inf_wh[0] - dst_w)/2)
        pad_r = int((self.inf_wh[0] - dst_w) - pad_l)
        pad_t = int((self.inf_wh[1] - dst_h)/2)
        pad_b = int((self.inf_wh[1] - dst_h) - pad_t)
        
        padded_img = np.pad(
            resized_img,
            ((pad_t,pad_b),(pad_l,pad_r)),
            mode='reflect')
        
        roi = (pad_l, pad_t, pad_l + dst_w, pad_t + dst_h)

        return padded_img, r, roi
    
    def preprocess(self, data_dict):
        if 'bgr_img' in data_dict.keys():
            gray_img = cv2.cvtColor(data_dict['bgr_img'],cv2.COLOR_BGR2GRAY)
        if 'rgb_img' in data_dict.keys():
            gray_img = cv2.cvtColor(data_dict['rgb_img'],cv2.COLOR_RGB2GRAY)
        
        data_dict['input_gray_img'],data_dict['r'],data_dict['roi'] = self.rsz_padding(gray_img)
        data_dict['input_tensor'] = np.reshape(
            self.normalize(data_dict['input_gray_img']),
            (1,self.inf_wh[1],self.inf_wh[0]))    
        return 
    
    def postprocess(self, data_dict):
        cropped_tensor = data_dict['output_tensor'][
            0,0,
            data_dict['roi'][1]:data_dict['roi'][3],
            data_dict['roi'][0]:data_dict['roi'][2]]
        data_dict['output_img'] = self.denormlize(cropped_tensor)
        return
    
    def inference(self, data_dict):
        ort_input = {
            self.sess.get_inputs()[0].name: data_dict['input_tensor'][None, :, :, :]
        }
        ort_output= self.sess.run(None, input_feed=ort_input)
        data_dict['output_tensor']  = ort_output[0]
        return

def process_dir(src_dir, dst_dir):
    os.makedirs(dst_dir,exist_ok=True)
    skechlizer =  OnnxSimpleGrayProcess(
        onnx_model_fn='./prototype.onnx',
        wh=(768,768),
        mean=127.5,
        std=127.5,
        gpu_id=3)

    for fn in tqdm(os.listdir(src_dir)):
        dst_fn = os.path.join(dst_dir,f'p2p_{fn}')
        src_fn = os.path.join(src_dir,fn)
        src_data_dict = {
            'bgr_img':cv2.imread(src_fn, cv2.IMREAD_COLOR + cv2.IMREAD_IGNORE_ORIENTATION)
        }
        out_dict = skechlizer(src_data_dict)

        cv2.imwrite(dst_fn,out_dict['output_img'])

    return


# srcimg = cv2.imread('./imgs/shutterstock_3826657.jpg')



# skechlizer =  OnnxSimpleGrayProcess(
#     onnx_model_fn='./prototype.onnx',
#     wh=(768,768),
#     mean=127.5,
#     std=127.5,
#     gpu_id=3)

# src_data_dict = {
#     'bgr_img':srcimg
# }

# out_dict = skechlizer(src_data_dict)

# cv2.imwrite('./imgs/res.png',out_dict['output_img'])


src_dir = '/media/112new_sde/CommonDatasets/sketchlize/ori'
dst_dir = '/media/112new_sde/CommonDatasets/sketchlize/p2p_res'

process_dir(src_dir, dst_dir)