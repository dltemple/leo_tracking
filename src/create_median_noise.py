from PIL import Image
import numpy as np
import os

import random
from joblib import Parallel, delayed
import multiprocessing


class Processor:
    def __call__(self, file):
        current_image = Image.open(file).convert('L')
        return np.array(current_image, dtype=np.uint8)


def extract_median_noise(filelist):
    nfiles = len(filelist)
    final_image = Image.open(filelist[0])
    im = np.zeros([final_image.size[1], final_image.size[0], nfiles], dtype=np.float32)
    for fidx, file in enumerate(filelist):
        print('Loading Image {0:4d} out of {1:4d}'.format(fidx, nfiles))
        current_image = Image.open(file).convert('L')
        im[:, :, fidx] = np.array(current_image, dtype=np.float32)

    median = np.median(im, -1, keepdims=True)


def extract_median_single_image(file):

    return median


if __name__ == '__main__':

    # median = np.load('median_array.npy')

    base_path = os.path.dirname(os.path.realpath(__file__)).replace('src', '')
    image_folder = os.path.join(base_path, 'data/2020-03-06/original/set2')

    file_list = os.listdir(image_folder)
    file_list_img = [os.path.join(image_folder, x)
                 for x in file_list if x.endswith(('.nef', 'nef'))]

    final_image = Image.open(file_list_img[0])
    images = random.sample(file_list_img, len(file_list_img))

    im = np.zeros([final_image.size[1], final_image.size[0], len(images)], dtype=np.uint8)

    proc = Processor()
    pool = multiprocessing.Pool()
    results = pool.map(proc, images)

    for ridx, result in enumerate(results):
        im[:, :, ridx] = result

    print('Calcuating Median of Images')
    median = np.median(im, -1, keepdims=True).astype(np.uint8)
    print('Saving Median Array')
    np.save('median_array_monochrome', median.reshape([4000, 6000]))

    print('Saving Median Image')
    temp_image = Image.fromarray(np.uint8(median[:, :, 0].clip(0, 255)))
    temp_image.save('median_image_raw.png', 'PNG')

    exit()