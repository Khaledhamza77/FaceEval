import math
import numpy as np
from .pyramid import PyramidAnalysis


class ImageQuality:
    def __init__(
        self, d_t=0.7, b_t=0.7, l_bin_size=30,
        v_r=2, h_r=2,
        box_l_r=0.9, box_s_r=0.01,
        occ_range=30, occ_t=0.8,
        testing=False
    ):
        self.dark_threshold = d_t
        self.bright_threshold = b_t
        self.lt_range = l_bin_size

        self.vertical_ratio = v_r
        self.horizontal_ratio = h_r

        self.bbox_large_ratio = box_l_r
        self.bbox_small_ratio = box_s_r

        self.occ_range = occ_range
        self.occ_threshold = occ_t

        self.testing = testing

    def bbox_ratio(self, img, bbox):
        area_box = abs(bbox[2] - bbox[0]) * abs(bbox[3] - bbox[1])
        area_img = img.shape[0] * img.shape[1]
        if area_box / area_img >= self.bbox_large_ratio:
            return 'Inappropriate Bounding Box: Large'
        elif area_box / area_img <= self.bbox_small_ratio:
            return 'Inappropriate Bounding Box: Small'
        else:
            out_x_min = max(0, -bbox[0])  # Portion left of the image
            out_y_min = max(0, -bbox[1])  # Portion above the image
            out_x_max = max(0, bbox[2] - img.shape[1])  # Portion right of the image
            out_y_max = max(0, bbox[3] - img.shape[0])  # Portion below the image

            out_w = (bbox[2] - bbox[0]) - (max(0, bbox[0]) - min(img.shape[1], bbox[2]))
            out_h = (bbox[3] - bbox[1]) - (max(0, bbox[1]) - min(img.shape[0], bbox[3]))

            out_of_bounds_area = max(0, out_w * out_y_min) + max(0, out_w * out_y_max) + max(0, out_h * out_x_min) + max(0, out_h * out_x_max)
            outside_ratio = out_of_bounds_area / area_box
            if outside_ratio > 0.5:
                if self.testing:
                    print(outside_ratio)
                return 'Inappropriate Bounding Box: Outside Image'
            else:
                return 'PASS'

    def side(self, point1, point2):
        return np.hypot(point1[0] - point2[0], point1[1] - point2[1])

    def pose(self, kps):
        a = self.side(kps[0], kps[2])
        b = self.side(kps[2], kps[3])
        c = self.side(kps[0], kps[3])
        s = (a + b + c) / 2
        A1 = math.sqrt(s * (s - a) * (s - b) * (s - c))
        del a, b, c, s

        a = self.side(kps[1], kps[2])
        b = self.side(kps[2], kps[4])
        c = self.side(kps[1], kps[4])
        s = (a + b + c) / 2
        A2 = math.sqrt(s * (s - a) * (s - b) * (s - c))

        if A2 == 0:
            A2 = 0.0000000001
        ratio11 = A1 / A2

        if A1 == 0:
            A1 = 0.0000000001
        ratio12 = A2 / A1
        del a, b, c, s, A1, A2

        a = self.side(kps[2], kps[3])
        b = self.side(kps[3], kps[4])
        c = self.side(kps[4], kps[2])
        s = (a + b + c) / 2
        A1 = math.sqrt(s * (s - a) * (s - b) * (s - c))
        del a, b, c, s

        a = self.side(kps[0], kps[1])
        b = self.side(kps[1], kps[2])
        c = self.side(kps[2], kps[0])
        s = (a + b + c) / 2
        A2 = math.sqrt(s * (s - a) * (s - b) * (s - c))

        if A2 == 0:
            A2 = 0.0000000001
        ratio21 = A1 / A2

        if A1 == 0:
            A1 = 0.0000000001
        ratio22 = A2 / A1
        del a, b, c, s, A1, A2
        if ((ratio11 >= self.horizontal_ratio) or (ratio12 >= self.horizontal_ratio)):
            return 'Bad Pose: Horizontal'
        elif ((ratio21 >= self.vertical_ratio) or (ratio22 >= self.vertical_ratio)):
            return 'Bad Pose: Vertical'
        else:
            return 'PASS'

    def router(self, i, img, aimg, kps, box):
        if i == 0:
            return self.bbox_ratio(img, box)
        elif i == 1:
            return self.pose(kps)
        else:
            return PyramidAnalysis(
                img=img,
                aimg=aimg,
                bbox=box,
                kps=kps,
                oc_threshold=self.occ_threshold,
                oc_range=self.occ_range,
                lt_range=self.lt_range,
                d_t=self.dark_threshold,
                b_t=self.bright_threshold,
                testing=self.testing
            ).run()

    def quality_checks(self, img, aimg, kps, box):
        for i in range(3):
            result = self.router(i, img, aimg, kps, box)
            if result != 'PASS':
                return result
        return 'PASS'