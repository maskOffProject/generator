import cv2
import random
import numpy as np


def flip(img):
    return cv2.flip(img, random.choice([-1, 0, 1]))


def zoom(img):
    h, w = img.shape[:2]
    scalar = random.uniform(0.4, 0.9)
    nh, nw = int(scalar * h), int(scalar * w)
    dh, dw = h - nh, w - nw
    zimg = img[dh // 2:nh + dh // 2, dw // 2:nw + dw // 2]
    if zimg.any():
        zimg = cv2.resize(zimg, (w, h))
    return zimg


def horizontal_shift(img):
    ratio = random.uniform(0, 1)
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = w * ratio
    ratio_img = []
    if ratio > 0:
        ratio_img = img[:, :int(w - to_shift), :]
    if ratio < 0:
        ratio_img = img[:, int(-1 * to_shift):, :]

    if ratio_img.any():
        img = cv2.resize(ratio_img, (h, w), interpolation=cv2.INTER_CUBIC)
    return img


def vertical_shift(img):
    ratio = random.uniform(0, 1)
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = h * ratio
    ratio_img = []
    if ratio > 0:
        ratio_img = img[:int(h - to_shift), :, :]
    if ratio < 0:
        ratio_img = img[int(-1 * to_shift):, :, :]

    if ratio_img.any():
        img = cv2.resize(ratio_img, (h, w), interpolation=cv2.INTER_CUBIC)
    return img


def rotation(img):
    angle = random.randint(1, 90)
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w / 2), int(h / 2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img


aug_mapper = {
    1: flip,
    2: zoom,
    3: horizontal_shift,
    4: vertical_shift,
    5: rotation,
}


def get_aug_img(img):
    choice = random.randint(0, 5)
    aug_chosen = aug_mapper.get(choice)
    if aug_chosen:
        return aug_chosen(img)
    return img
