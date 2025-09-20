import cv2
import numpy as np
from PIL import Image


class PyramidAnalysis:
    def __init__(self, img, aimg, bbox, kps, oc_threshold,
                 oc_range, lt_range, d_t, b_t, **kwargs):
        self.bbox = bbox
        self.test = kwargs.get('testing', False)
        self.img, self.aimg = self.get_intrinsic(img, aimg, kps)
        self.oc_threshold = oc_threshold
        self.oc_range = oc_range
        self.dark_threshold = d_t
        self.bright_threshold = b_t
        self.lt_range = lt_range

        self.vertical_r = 25
        self.horizontal_r = 32

    def clamp(self, val, max_val):
        return int(max(0, min(val, max_val)))

    def get_intrinsic(self, img, aimg, kps):
        img_h, img_w = img.shape[:2]
        x_min = self.clamp(self.bbox[0], img_w)
        y_min = self.clamp(self.bbox[1], img_h)
        x_max = self.clamp(self.bbox[2], img_w)
        y_max = self.clamp(self.bbox[3], img_h)
        cropped_img = img[y_min:y_max, x_min:x_max]
        cropped_h, cropped_w = cropped_img.shape[:2]
        crrs_img = cv2.resize(cropped_img, (256, 256))

        self.kps = [
            (int((x - x_min) * (256 / cropped_w)), int((y - y_min) * (256 / cropped_h))) for (x, y) in kps
        ]

        intrinsic = cv2.bilateralFilter(
            cv2.cvtColor(crrs_img, cv2.COLOR_BGR2GRAY),
            d=10, sigmaColor=100, sigmaSpace=100
        )
        if self.test:
            to_show = intrinsic.copy()
            for (x, y) in self.kps:
                cv2.circle(to_show, (x, y), radius=5,
                           color=(0, 0, 0), thickness=-1)

            Image.fromarray(to_show).show()

        aintrinsic = cv2.bilateralFilter(
            cv2.cvtColor(aimg, cv2.COLOR_BGR2GRAY),
            d=5, sigmaColor=50, sigmaSpace=50
        )

        return intrinsic, aintrinsic

    def crop_landmarks(self):
        cropped_imgs = []
        img_h, img_w = self.img.shape[:2]

        # Left Eye
        x, y = self.kps[0]
        x_min = self.clamp(x - self.horizontal_r, img_w)
        y_min = self.clamp(y - self.vertical_r, img_h)
        x_max = self.clamp(x + self.horizontal_r, img_w)
        y_max = self.clamp(y + self.vertical_r, img_h)
        cropped_imgs.append(('Left Eye', self.img[y_min:y_max, x_min:x_max]))

        # Right Eye
        x, y = self.kps[1]
        x_min = self.clamp(x - self.horizontal_r, img_w)
        y_min = self.clamp(y - self.vertical_r, img_h)
        x_max = self.clamp(x + self.horizontal_r, img_w)
        y_max = self.clamp(y + self.vertical_r, img_h)
        cropped_imgs.append(('Right Eye', self.img[y_min:y_max, x_min:x_max]))

        # Nose
        x, y = self.kps[2]
        x_min = self.clamp(x - self.horizontal_r, img_w)
        y_min = self.clamp(y - int(round(1.5 * self.vertical_r)), img_h)
        x_max = self.clamp(x + self.horizontal_r, img_w)
        y_max = self.clamp(y + int(round(0.5 * self.vertical_r)), img_h)
        cropped_imgs.append(('Nose', self.img[y_min:y_max, x_min:x_max]))

        # Lips
        llip, rlip = self.kps[3], self.kps[4]
        x1, y1, x2, y2 = llip[0], llip[1], rlip[0], rlip[1]
        y_min = self.clamp(min(y1, y2) - self.vertical_r, img_h)
        y_max = self.clamp(max(y1, y2) + self.vertical_r, img_h)
        x_min = self.clamp(x1, img_w)
        x_max = self.clamp(x2, img_w)
        cropped_imgs.append(('Lips', self.img[y_min:y_max, x_min:x_max]))

        return cropped_imgs

    def create_pyramid(self, img):
        h, w = img.shape[:2]
        num_ups = 3

        pyramid = [img]
        for i in range(num_ups):
            pyramid.append(cv2.pyrUp(pyramid[-1]))

        pyramid.reverse()
        return pyramid[:2]

    def single_occlusion_det(self, img):
        total_pixels = int(img.shape[0] * img.shape[1])
        threshold = total_pixels * self.oc_threshold

        start, end = 0, 0
        while end < 100:
            end = min(start + self.oc_range, 100)
            binary_mask = cv2.inRange(img, start, end)
            num_labels, labels = cv2.connectedComponents(binary_mask)
            largest_connected_dark = 0

            for label in range(1, num_labels):
                dummy = np.sum(labels == label)
                if dummy > largest_connected_dark:
                    largest_connected_dark = dummy

            if largest_connected_dark >= threshold:
                if self.test:
                    Image.fromarray(img).show()
                    print(f"[{start}, {end}]")
                    print('Largest Connected: ', largest_connected_dark)
                    print('Threshold: ', threshold)
                return 'Occlusion'

            start += self.oc_range // 2

        return 'PASS'

    def occlusion_detection(self):
        results = []
        cropped = self.crop_landmarks()

        for landmark, img in cropped:
            counter = 0
            if img.size > 0:
                pyramid = self.create_pyramid(img)

                for i, level in enumerate(pyramid):
                    if self.single_occlusion_det(level) == 'Occlusion':
                        counter = len(pyramid) - i
                        results.append(landmark)
                        break

            if counter < len(pyramid) - 1:
                results.append('PASS')

        return results

    def single_lighting_eval(self, img):
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        dark_pixels = np.sum(hist[:self.lt_range])
        bright_pixels = np.sum(hist[256 - self.lt_range:])
        total_pixels = int(img.shape[0] * img.shape[1])

        if (dark_pixels / total_pixels) > self.dark_threshold:
            if self.test:
                Image.fromarray(img).show()
                print('dark pixels: ', dark_pixels)
                print('threshold: ', total_pixels * self.dark_threshold)
            return 'Dark'
        elif (bright_pixels / total_pixels) > self.bright_threshold:
            if self.test:
                Image.fromarray(img).show()
                print('bright pixels: ', bright_pixels)
                print('threshold: ', total_pixels * self.bright_threshold)
            return 'Bright'
        else:
            return 'PASS'

    def lighting_evaluation(self):
        pyramid = self.create_pyramid(self.aimg)
        Image.fromarray(self.aimg).show() if self.test else None

        for level in pyramid:
            issue = self.single_lighting_eval(level)
            if issue == 'Dark':
                return 'Bad Lighting: Dark'
            elif issue == 'Bright':
                return 'Bad Lighting: Bright'

        return 'PASS'

    def run(self):
        lt_result = self.lighting_evaluation()
        if lt_result.startswith('Bad'):
            return lt_result

        oc_results = self.occlusion_detection()
        result, counter = '', 0
        for landmark in oc_results:
            if landmark != 'PASS':
                counter += 1
                result += f'{landmark} '

        if counter < 2:
            return 'PASS'
        else:
            return f'Dimly Lit: {result}'.strip()