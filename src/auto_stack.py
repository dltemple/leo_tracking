import os
from time import time

import cv2
import numpy as np

import sys
import datetime
import imageio
from pprint import pprint
import time
import datetime

from skimage import transform

from PIL import Image, ImageChops, ImageEnhance
from astropy.io import fits

import astropy.io.fits as pyfits
import skimage.morphology as morph
import skimage.exposure as skie

import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)


def convert_to_fits(filelist):
    for file in filelist:
        image = Image.open(file).convert('L')
        xsize, ysize = image.size
        print("Image size: {} x {}".format(xsize, ysize))

        nxsize, nysize = xsize // 10, ysize // 10

        c1 = image.split()
        c1_data = np.array(c1[0].getdata())  # data is now an array of length ysize*xsize

        c1_data = c1_data.reshape(ysize, xsize)

        c1 = fits.PrimaryHDU(data=c1_data)
        c1.writeto(file.replace('.jpeg', '.fits'))

        image = image.resize((nxsize, nysize))
        image.save(file.replace('.jpeg', '_s.jpeg'), "JPEG")


def create_gif(filenames, duration, output_file=None):
    images = []
    for filename in filenames:
        image = imageio.imread(filename)
        small_grey = transform.rescale(image, scale=0.1)
        images.append(small_grey)
    if output_file is None:
        output_file = 'Gif-%s.gif' % datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S')

    imageio.mimsave(output_file, images, duration=duration)


# Align and stack images with ECC method
# Slower but more accurate
def stackImagesECC(file_list):
    M = np.eye(3, 3, dtype=np.float32)

    first_image = None
    stacked_image = None

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)

    for file in file_list:
        image = cv2.imread(file, 1).astype(np.float32) / 255
        print(file)
        if first_image is None:
            # convert to gray scale floating point image
            first_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            stacked_image = image
        else:
            # Estimate perspective transform
            s, M = cv2.findTransformECC(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), first_image, M, cv2.MOTION_HOMOGRAPHY, criteria, None, 5)
            w, h, _ = image.shape
            # Align image to first image
            image = cv2.warpPerspective(image, M, (h, w))
            stacked_image += image

    stacked_image /= len(file_list)
    stacked_image = (stacked_image * 255).astype(np.uint8)
    return stacked_image


# Align and stack images by matching ORB keypoints
# Faster but less accurate
def stackImagesKeypointMatching(file_list):
    orb = cv2.ORB_create()

    # disable OpenCL to because of bug in ORB in OpenCV 3.1
    cv2.ocl.setUseOpenCL(False)

    stacked_image = None
    first_image = None
    first_kp = None
    first_des = None
    for file in file_list:
        print(file)
        image = cv2.imread(file, 1)
        imageF = image.astype(np.float32) / 255

        # compute the descriptors with ORB
        kp = orb.detect(image, None)
        kp, des = orb.compute(image, kp)

        # create BFMatcher object
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        if first_image is None:
            # Save keypoints for first image
            stacked_image = imageF
            first_image = image
            first_kp = kp
            first_des = des
        else:
            # Find matches and sort them in the order of their distance
            matches = matcher.match(first_des, des)
            matches = sorted(matches, key=lambda x: x.distance)

            src_pts = np.float32(
                [first_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate perspective transformation
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            w, h, _ = imageF.shape
            imageF = cv2.warpPerspective(imageF, M, (h, w))
            stacked_image += imageF

    stacked_image /= len(file_list)
    stacked_image = (stacked_image * 255).astype(np.uint8)
    return stacked_image


def convert_to_grayscale(filelist):
    for file in filelist:
        image = cv2.imread(file, 1).astype(np.float32)
        image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(file.replace('.jpeg', '_gray.jpeg')), image2)


def remove_median_noise(filelist):
    final_image = Image.open(filelist[0])
    im = np.zeros([final_image.size[1], final_image.size[0], len(filelist)], dtype=np.float32)
    for fidx, file in enumerate(filelist):
        print('Loading Image {0:4d}'.format(fidx))
        current_image = Image.open(file).convert('L')
        im[:, :, fidx] = np.array(current_image, dtype=np.float32)

    im2 = im - np.median(im, -1, keepdims=True)

    for fidx, file in enumerate(filelist):
        temp_image = Image.fromarray(np.uint8(im2[:, :, fidx].clip(0, 255)))
        temp_image.save(file, 'JPEG')


def enhance_contrast(filelist):
    for fidx, file in enumerate(filelist):
        temp_image = Image.open(file)
        im_arr = np.array(temp_image, dtype=np.float32)
        im_arr = gamma_correction(im_arr)
        image_out = Image.fromarray(np.uint8(im_arr.clip(0, 255)))
        image_out.save(file.replace('.jpeg', '_c.jpeg'), 'JPEG')


def gamma_correction(image0):
    min_snr = -3
    max_snr = 100
    infl_snr = 2.0
    yfrac = 0.1
    gamma = 0.45

    u = np.median(image0)
    s = np.sqrt(np.mean((image0 - u) ** 2))
    if s == 0:
        s = 1

    pflat = image0.flatten()
    pflat = (pflat - u) / s
    za = pflat < min_snr
    zb = np.logical_and(pflat >= min_snr, pflat < infl_snr)
    zc = np.logical_and(pflat >= infl_snr, pflat <= max_snr)
    zd = pflat > max_snr

    pflat[za] = 0
    pflat[zb] = yfrac * ((pflat[zb] - min_snr) / (infl_snr - min_snr)) ** (1 / gamma)
    pflat[zc] = yfrac + (1 - yfrac) * ((pflat[zc] - infl_snr) / (max_snr - infl_snr)) ** gamma
    pflat[zd] = 1

    image = (np.reshape(pflat, image0.shape))
    if np.max(image0 > 255):
        image = (65535 * ((image - image.min()) / image.ptp())).astype(np.uint16)
    else:
        image = (255 * ((image - image.min()) / image.ptp())).astype(np.uint8)

    return image


def stack_images_max(file_list):

    final_image = Image.open(file_list[0])
    im = np.array(final_image, dtype=np.float32)
    for file in file_list:
        current_image = Image.open(file)
        final_image = ImageChops.lighter(final_image, current_image)
        # im += np.array(current_image, dtype=np.float32)

    # im /= len(file_list) * 0.25
    # final_image = Image.fromarray(np.uint8(im.clip(0, 255)))
    final_image.save('all_averaged.jpg', 'JPEG')


def process_fits(fits_file_list):
    for infits in fits_file_list:
        img = pyfits.getdata(infits)
        limg = np.arcsinh(img)
        limg /= limg.max()
        low = np.percentile(limg, 0.25)
        high = np.percentile(limg, 99.5)
        opt_img = skie.exposure.rescale_intensity(limg, in_range=(low, high))

        opt_img
    # lm = morph.local_maxima(limg)
    # x1, y1 = np.where(lm.T == True)
    # v = limg[(y1, x1)]
    # lim = 0.7
    # x2, y2 = x1[v > lim], y1[v > lim]

# ===== MAIN =====
# Read all files in directory


if __name__ == '__main__':

    base_path = os.path.dirname(os.path.realpath(__file__)).replace('src', '')
    image_folder = os.path.join(base_path, 'data/two_sats')

    file_list = os.listdir(image_folder)
    file_list_img = [os.path.join(image_folder, x)
                 for x in file_list if x.endswith(('.jpg', 'jpeg'))]

    file_list_fits = [os.path.join(image_folder, x)
                     for x in file_list if x.endswith(('.fits'))]

    # process_fits(file_list_fits)
    remove_median_noise(file_list_img)
    enhance_contrast(file_list_img)
    # convert_to_grayscale(file_list_img)
    # stack_images_max(file_list)

    # convert_to_fits(file_list)
    create_gif(file_list, duration=0.2, output_file=output_file)

    exit()
