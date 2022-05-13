import cv2
import random


class Daugmentation:
    def flip(img, t):
        if t[0] == 0:
            return img
        else:
            return cv2.flip(img, t[1])

    def zoom(img, t):
        if t[2] == 0:
            return img
        else:
            h, w = img.shape[:2]
            nh, nw = int(t[3] * h), int(t[3] * w)
            dh, dw = h - nh, w - nw
            zimg = img[dh // 2:nh + dh // 2, dw // 2:nw + dw // 2]
            zimg = cv2.resize(zimg, (w, h))
            return zimg

    def get_ts(batch_size):
        return [[random.choice([0, 1]), random.choice([-1, 0, 1]), random.randint(0, 2), random.uniform(0.4, 0.9)] for i
                in range(batch_size)]

    def aug(img, t):
        img = Daugmentation.flip(img, t)
        return img
