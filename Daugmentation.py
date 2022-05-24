import cv2
import random
import numpy as np


def flip(img):
    value = random.choice([-1, 0, 1])
    flipped_img = cv2.flip(img, value)
    return flipped_img


def zoom(img):
    h, w = img.shape[:2]
    scalar = random.uniform(0.4, 0.9)
    nh, nw = int(scalar * h), int(scalar * w)
    dh, dw = h - nh, w - nw
    zimg = img[dh // 2:nh + dh // 2, dw // 2:nw + dw // 2]
    zimg = cv2.resize(zimg, (w, h))
    return zimg


def horizontal_shift(img):
    ratio = random.uniform(0, 1)
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = w * ratio
    if ratio > 0:
        img = img[:, :int(w - to_shift), :]
    if ratio < 0:
        img = img[:, int(-1 * to_shift):, :]
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img


def vertical_shift(img):
    ratio = random.uniform(0, 1)
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = h * ratio
    if ratio > 0:
        img = img[:int(h - to_shift), :, :]
    if ratio < 0:
        img = img[int(-1 * to_shift):, :, :]
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img


def channel_shift(img):
    value = random.randint(0, 90)
    value = int(random.uniform(-value, value))
    img = img + value
    img[:, :, :][img[:, :, :] > 255] = 255
    img[:, :, :][img[:, :, :] < 0] = 0
    img = img.astype(np.uint8)
    return img


def rotation(img):
    angle = random.randint(0, 90)
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
    5: channel_shift,
    6: rotation,
}


def get_aug_img(img):
    choice = random.randint(0, 6)
    aug_chosen = aug_mapper.get(choice)
    if aug_chosen:
        return aug_chosen(img)
    return img
