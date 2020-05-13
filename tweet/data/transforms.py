import cv2
import io
import os
import os.path as osp
from PIL import Image
import numpy as np
import random
import torchvision
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import math

## random crop
def random_crop(img, crop_ratio=0.05):
    w, h = img.size
    crop_w, crop_h = crop_ratio * w, crop_ratio * h
    ratios = list(np.linspace(0.0, 1.0, 10))
    choices = [random.choice(ratios) for i in range(4)]

    img = img.crop((crop_w*choices[0], crop_h*choices[1], w-crop_w*choices[2], h-crop_h*choices[3]))
    return img

## random occlusion
def random_occlusion(img, occlusion_threshold=1.0):
    if random.random() > occlusion_threshold:
        img = random_vertical_crop(img) if random.random() > 0.5 else random_erase(img)
    return img

## vertical crop
def random_vertical_crop(img):
    w, h = img.size
    rate = random.uniform(0.3, 0.7)
    if random.random() > 0.5:
        img = img.crop((0,0,rate*w,h))
    else:
        img = img.crop((rate*w,0,w,h))

    return img

## random erase
def random_erase(img):
    w, h = img.size
    r_w, r_h = random.uniform(0.1, 0.5), random.uniform(0.1, 0.5)
    Rc, Gc, Bc = random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)
    if random.random() > 0.5:
        occlusion = img.crop((0, (1-r_h)*h, r_w*w, h))
    else:
        occlusion = img.crop(((1-r_w)*w, (1-r_h)*h, w, h))
    I = Image.new('RGB', occlusion.size, (Rc, Gc, Bc))
    img.paste(I, (0, int((1-r_h)*h)))

    return img

## resize
def cv2_resize(img, width, height):
    return cv2.resize(img, (width, height))

## resize with padding
def cv2_resize_padding(img, width, height, padding):
    img = np.array(img)
    if padding not in ['edge', 'const']:
        return cv2_resize(img, width, height)
    else:
        w, h = img.shape[:2]
        new_w = min(int(h / height * w), width)
        img = cv2.resize(img, (new_w, height))
        if padding == 'edge':
            img = np.pad(img, ((0, 0), (0, width-new_w), (0, 0)), 'edge')
        else:
            img = np.pad(img, ((0, 0), (0, width-new_w), (0, 0)), 'constant', constant_values=(0, 0))
    return img

def image_transpose(img, angle):
    if angle not in [90, 180, 270]:
        print('transpose angle must be in [90, 180, 270]')
        return img
    else:
        if angle == 90:
            return img.transpose(Image.ROTATE_270)
        elif angle == 180:
            return img.transpose(Image.ROTATE_180)
        elif angle == 270:
            return img.transpose(Image.ROTATE_90)

## auto augmentation
class ImageNetPolicy(object):
    def __init__(self, keep_prob=0.5):
        self.keep_prob = keep_prob
        self.range = {
            "shearX": np.linspace(-0.1, 0.1, 10),
            "shearY": np.linspace(-0.1, 0.1, 10),
            "rotate": np.linspace(-10, 10, 10),
#            "color": np.linspace(0.6, 1.5, 10),
#            "posterize": np.round(np.linspace(4, 8, 10), 0).astype(np.int),
#            "contrast": np.linspace(0.6, 1.5, 10),
#            "sharpness": np.linspace(0.1, 1.9, 10),
#            "brightness": np.linspace(0.5, 1.4, 10),
#            "autocontrast": [0] * 10,
#            "equalize": [0] * 10,
#            "gauss_blur": np.linspace(0.5, 1.0, 10),
#            "motion_blur": [1, 1, 1 ,1, 2, 2, 3, 3, 4, 5],
#            "detail": [0] * 10
        }

    def func(self, op, img, magnitude):
        if op== "shearX": return self.shear(img, magnitude * 180, direction="x")
        elif op== "shearY": return self.shear(img, magnitude * 180, direction="y")
        elif op== "rotate": return img.rotate(magnitude)
        elif op=="color": return ImageEnhance.Color(img).enhance(magnitude)
        elif op=="posterize": return ImageOps.posterize(img, magnitude)
        elif op=="contrast": return ImageEnhance.Contrast(img).enhance(magnitude)
        elif op=="sharpness": return ImageEnhance.Sharpness(img).enhance(magnitude)
        elif op=="brightness": return ImageEnhance.Brightness(img).enhance(magnitude)
        elif op=="autocontrast": return ImageOps.autocontrast(img)
        elif op=="equalize": return ImageOps.equalize(img)
        elif op=="gauss_blur": return img.filter(ImageFilter.GaussianBlur(radius=magnitude))
        elif op=="motion_blur": return self.motion_blur(img, magnitude)
        elif op=="detail": return img.filter(ImageFilter.DETAIL)
        else: print('error ops')

    def __call__(self, img):
        if random.random() < self.keep_prob:
            return img
        else:
            rand = np.random.randint(0, 10, 2)
            policies = random.sample(list(self.range.keys()), 2)
            img = self.func(policies[0], img, self.range[policies[0]][rand[0]])
            if random.random() < 0.5:
                img = self.func(policies[1], img, self.range[policies[1]][rand[1]])
            return img

    def motion_blur(self, image, degree=6, angle=45):
        image = np.array(image)
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)

        # convert to uint8
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = torchvision.transforms.ToPILImage()(blurred)
        return blurred

    def shear(self, img, angle_to_shear, direction="x"):
        width, height = img.size
        phi = math.tan(math.radians(angle_to_shear))

        if direction=="x":
            shift_in_pixels = phi * height

            if shift_in_pixels > 0:
                shift_in_pixels = math.ceil(shift_in_pixels)
            else:
                shift_in_pixels = math.floor(shift_in_pixels)

            matrix_offset = shift_in_pixels
            if angle_to_shear <= 0:
                shift_in_pixels = abs(shift_in_pixels)
                matrix_offset = 0
                phi = abs(phi) * -1

            transform_matrix = (1, phi, -matrix_offset, 0, 1, 0)

            img = img.transform((int(round(width + shift_in_pixels)), height),
                                    Image.AFFINE,
                                    transform_matrix,
                                    Image.BICUBIC, fillcolor=(128, 128, 128))

            return img.resize((width, height), resample=Image.BICUBIC)

        elif direction == "y":
            shift_in_pixels = phi * width

            matrix_offset = shift_in_pixels
            if angle_to_shear <= 0:
                shift_in_pixels = abs(shift_in_pixels)
                matrix_offset = 0
                phi = abs(phi) * -1

            transform_matrix = (1, 0, 0, phi, 1, -matrix_offset)

            image = img.transform((width, int(round(height + shift_in_pixels))),
                                    Image.AFFINE,
                                    transform_matrix,
                                    Image.BICUBIC, fillcolor=(128, 128, 128))

            return image.resize((width, height), resample=Image.BICUBIC)
