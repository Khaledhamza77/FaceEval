import cv2
import glob
import numpy as np
import onnxruntime
import os.path as osp
from PIL import Image
from .quality.evaluate import ImageQuality
from insightface.utils import face_align
from insightface.model_zoo import model_zoo
from insightface.utils import DEFAULT_MP_NAME, ensure_available


class FaceEvaluator:
    def __init__(self, root='~/.insightface',
                 det_size=(160, 160), det_thresh=0.65,
                 show_img=True,
                 d_t=0.7, b_t=0.7, l_bin_size=30,
                 v_r=4, h_r=6,
                 box_l_r=0.9, box_s_r=0.2,
                 occ_range=30, occ_t=0.8,
                 testing=False, **kwargs):

        onnxruntime.set_default_logger_severity(3)
        model_dir = ensure_available('models', DEFAULT_MP_NAME, root=root)
        onnx_files = glob.glob(osp.join(model_dir, '*.onnx'))
        onnx_files = sorted(onnx_files)
        self.detector = model_zoo.get_model(onnx_files[2], **kwargs)
        device_id = kwargs.get('device_id', 0)
        self.detector.prepare(
            device_id,
            input_size=det_size,
            det_thresh=det_thresh
        )
        self.quality = ImageQuality(
            d_t=d_t,
            b_t=b_t,
            l_bin_size=l_bin_size,
            v_r=v_r, h_r=h_r,
            box_l_r=box_l_r,
            box_s_r=box_s_r,
            occ_range=occ_range,
            occ_t=occ_t,
            testing=testing
        )
        self.show_img = show_img

    def extract_face(self, img_path):
        img = cv2.imread(img_path)
        bboxes, kpss = self.detector.detect(img, max_num=0, metric='default')
        count = bboxes.shape[0]
        if count == 1:
            kps = None
            if kpss is not None:
                kps = kpss[0]

            aimg = face_align.norm_crop(img, landmark=kps,
                                        image_size=self.arcface.input_size[0])
            if self.show_img:
                dimg = img.copy()
                box = bboxes[0].astype(int)
                color = (0, 0, 255)
                cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]),
                              color, 2)
                kpss = kps.astype(int)
                for i in range(kpss.shape[0]):
                    color = (0, 0, 255)
                    if i == 0 or i == 3:
                        color = (0, 255, 0)
                    cv2.circle(dimg, (kpss[i][0], kpss[i][1]), 1, color, 2)
                Image.fromarray(dimg[:, :, ::-1]).show()
            return img, aimg, kps.astype(int), bboxes[0]
        else:
            return count, None, None, None
    
    def run(self, img_path):
        img, aimg, kps, box = self.extract_face(img_path)
        return self.quality.quality_checks(img=img, aimg=aimg, kps=kps, box=box)