from torchvision.transforms import *
from PIL import Image
import random
import numpy as np
import scipy.ndimage as ndi
from math import floor


class RandomZoom(object):
    def __init__(self, prob=0.0, zoom_range=[1, 1]):
        self.prob = prob
        self.zoom_range = zoom_range

    def __call__(self, image):
        if random.uniform(0, 1) > self.prob:
            return image
        else:
            w, h = image.size
            factor = random.uniform(self.zoom_range[0], self.zoom_range[1])
            image_zoomed = image.resize((int(round(image.size[0] * factor)),
                                         int(round(image.size[1] * factor))),
                                        resample=Image.BICUBIC)
            w_zoomed, h_zoomed = image_zoomed.size

            return image_zoomed.crop((floor((float(w_zoomed) / 2) - (float(w) / 2)),
                                      floor((float(h_zoomed) / 2) -
                                            (float(h) / 2)),
                                      floor((float(w_zoomed) / 2) +
                                            (float(w) / 2)),
                                      floor((float(h_zoomed) / 2) + (float(h) / 2))))


class RandomStretch(object):
    def __init__(self, prob=0.0, stretch_range=[1, 1]):
        self.prob = prob
        self.stretch_range = stretch_range

    def __call__(self, image):
        if random.uniform(0, 1) > self.prob:
            return image
        else:
            w, h = image.size
            factor = random.uniform(
                self.stretch_range[0], self.stretch_range[1])
            if factor <= 1:
                image_stretched = image.resize((int(round(image.size[0] / factor)),
                                                int(round(image.size[1]))),
                                               resample=Image.BICUBIC)
            else:
                image_stretched = image.resize((int(round(image.size[0])),
                                                int(round(image.size[1] * factor))),
                                               resample=Image.BICUBIC)
            w_stretched, h_stretched = image_stretched.size

            return image_stretched.crop((floor((float(w_stretched) / 2) - (float(w) / 2)),
                                         floor((float(h_stretched) / 2) -
                                               (float(h) / 2)),
                                         floor((float(w_stretched) / 2) +
                                               (float(w) / 2)),
                                         floor((float(h_stretched) / 2) + (float(h) / 2))))


class RandomResize(object):
    def __init__(self, prob=0.0, resize_range=[1, 1]):
        self.prob = prob
        self.resize_range = resize_range

    def __call__(self, image):
        if random.uniform(0, 1) > self.prob:
            return image
        else:
            w, h = image.size
            factor_w = random.uniform(
                self.resize_range[0], self.resize_range[1])
            factor_h = random.uniform(
                self.resize_range[0], self.resize_range[1])
            image_resized = image.resize((int(round(image.size[0] * factor_w)),
                                          int(round(image.size[1] * factor_h))),
                                         resample=Image.BICUBIC)
            w_resized, h_resized = image_resized.size

            return image_resized.crop((floor((float(w_resized) / 2) - (float(w) / 2)),
                                       floor((float(h_resized) / 2) -
                                             (float(h) / 2)),
                                       floor((float(w_resized) / 2) +
                                             (float(w) / 2)),
                                       floor((float(h_resized) / 2) + (float(h) / 2))))


class RandomRotation(object):
    def __init__(self, prob=0.0, degree=[0, 0]):
        self.prob = prob
        self.degree = degree

    def __call__(self, image):
        if random.uniform(0, 1) > self.prob:
            return image
        else:
            rotate_degree = random.uniform(self.degree[0], self.degree[1])
            # return image.rotate(rotate_degree, filter='NEAREST')
            rotated_array = ndi.interpolation.rotate(
                image, rotate_degree, reshape=False, mode='nearest')
            rotated_img = Image.fromarray(rotated_array)

            return rotated_img


def RotateTensor(input, degree):
    transform = transforms.Compose([transforms.ToPILImage(),
                                    RandomRotation(prob=1.0, degree=degree),
                                    transforms.ToTensor()])
    return transform(input)


# def RelabeledRandomRotation(input, label, num_rotation_class=1, prob=0.0):
#     batch_size = label.size()[0]
#     interval = 360 / num_rotation_class
#     for i in range(batch_size):
#         if random.uniform(0, 1) <= prob:
#             rotation_class = random.randint(1, num_rotation_class)
#             label[i] = label[i] * num_rotation_class + rotation_class - 1
#             degree = (interval * (rotation_class - 1), interval * rotation_class)
#             rotated_image = RotateTensor(input[i], degree)
#             input[i] = rotated_image
#     return input, label

def RelabeledRandomRotation(input, label, num_rotation_class=1):
    input_new = input.clone()
    label_new = label.clone()
    batch_size = label.size()[0]
    interval = 360 / num_rotation_class
    for i in range(batch_size):
        rotation_class = random.randint(1, num_rotation_class)
        label_new[i] = label[i] * num_rotation_class + rotation_class - 1
        degree = [interval * (rotation_class - 1), interval * rotation_class]
        rotated_image = RotateTensor(input[i], degree)
        input_new[i] = rotated_image
    return input_new, label_new
