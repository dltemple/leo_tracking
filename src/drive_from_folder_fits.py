import os
import glob
import subprocess
import datetime
import time
import math

# import jdutil
# import gpstime

import cv2
# import threading
# from queue import Queue

import math
from time import sleep

import numpy as np

# import imageio
# from skimage import transform

import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.convolution import Gaussian2DKernel
from astropy.stats import sigma_clipped_stats, gaussian_fwhm_to_sigma
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

from classes.Camera import Camera
from classes.Conversions import dt2jd
from classes.LeoProcessor import LEOProcessor

from PIL import Image, ImageChops

# from photutils import find_peaks, detect_threshold, detect_sources, Background2D

# import astroalign as aa
# from astride import Streak

image_dir = r'C:\Users\dtemple\AppData\Local\cygwin_ansvr'
launch_bash = 'C:/Users/dtemple/AppData/Local/cygwin_ansvr/bin/bash.exe --login -i '
output_path = r'C:\Users\dtemple\AppData\Local\cygwin_ansvr\tmp'
pixel_cmd = '--scale-units arcsecperpix --scale-low 22 --scale-high 24 --overwrite --downsample 4 --no-plots --crpix-center --no-remove-lines'


def camera_capture_task(camera, save_dir):
    print('Capturing Image')
    shot_date = datetime.datetime.utcnow().strftime("%Y-%m-%d")  # This has been written to the while True loop.
    # save_location = createSaveFolder(os.path.join(save_dir, shot_date))
    time.sleep(1)
    # image_file = capture_image(save_location=save_location)
    # image_file = camera.capture_single_image(save_folder=save_location)
    return shot_date


def process_image_task(observer: LEOProcessor, file: str):
    print('Processing IMAGE FILE {}'.format(file))
    # image = Image.open(file).convert('L')  # converts to grayscale
    # image_arr = subtract_median(image, median_arr)  # subtracts median values for camera and lens
    # image_arr = gamma_correction(image_arr)
    #
    # temp_image = Image.fromarray(np.uint8(image_arr.clip(0, 255)))
    # temp_image.save(image_file, 'JPEG')


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    print('Lower :: {}  Upper :: {}'.format(lower, upper))

    return cv2.Canny(image, lower, upper)


if __name__ == '__main__':
    """
    Work flow::

    1) Capture images
    2) search for streaks
    3) if no streaks, return to 1)
    4) if streaks found, plate solve image
    5) find streak start, stop, and centroid
    6) convert coordinates to RA/DEC
    7) store obs 
    8) fit orbit
    9) compare orbit to TLEs
    10) associate

    # D5200 w/ 35mm
    # FOV :: 37.6 x 25.1 deg
    # pixel scale :: 22.6 arcsec/pixel

    # D5200 w/ 50mm
    # FOV ::
    # pixel scale :: 22.6 arcsec/pixel

    """

    ext = '.jpg'

    base_path = os.path.dirname(os.path.realpath(__file__)).replace('src', '')
    save_dir = os.path.join(base_path, 'data')

    image_dir = os.path.join(save_dir, '2020-02-23/original/sat2/')
    processed_dir = image_dir.replace('original', 'processed')

    observer = LEOProcessor(base_path)
    observer.image_dir = image_dir

    images = glob.glob(os.path.join(image_dir, '*' + ext))
    # processed_images = glob.glob(os.path.join(processed_dir, '*' + ext))

    observer.create_save_folder(processed_dir)
    bg_sub_dir = observer.create_save_folder(os.path.join(processed_dir, 'b'))
    enhanced_dir = observer.create_save_folder(os.path.join(processed_dir, 'c'))

    enhanced_images = glob.glob(os.path.join(enhanced_dir, '*' + '.jpg'))

    # stack_images_max(enhanced_images[1:7])
    # observer.create_gif(enhanced_images[1:7], output_file='D:/leo_tracking/results/chinese.gif')
    # exit()

    median_arr = np.load(os.path.join(base_path, 'src', 'median_array_monochrome.npy')).astype(np.float32)
    median_arr = cv2.resize(median_arr, observer.reduced_image_size, interpolation=cv2.INTER_AREA)

    prev_arr = None

    observer.median_image = median_arr

    # center crop params
    minx, maxx = 2000 - 256, 2000 + 256
    miny, maxy = 3556 - 256, 3556 + 256

    # out_array = np.zeros([512 * 512, len(images)-1])

    # with open('D:/leo_tracking/src/test_512_raw_starsub.ofp', 'wb') as f:
    for idx, image_file in enumerate(images[:]):

        print('{}'.format(image_file))

        output, err, wcs_file = observer.plate_solve(image_file)

        observer.pix2RD(wcs_file)

        observer.read_image(image_file)
        # observer.image = observer.image_original
        observer.subtract_median()

        if idx == 0:
            prev_aligned = np.zeros_like(observer.image)
            prev_aligned = cv2.resize(prev_aligned, observer.reduced_image_size, interpolation=cv2.INTER_AREA)
            cur_stars, pmedian, pstd = observer.find_stars(observer.image)
        else:
            t0 = time.time()
            print('Aligning Stars :: ...')
            prev_aligned, cur_stars, pmedian, pstd = observer.stack_images_keypoints(prev_stars)
            print('Aligning Stars :: Time {0:4.4f}'.format(time.time() - t0))

        observer.pmedian, observer.pstd = pmedian, pstd
        bkg_subtracted_image = observer.subtract_image(prev_aligned)

        # observer.write_image(bkg_subtracted_image, bg_sub_dir)

        # corrected_image = observer.gamma_correction(bkg_subtracted_image, gamma=0.25, min_snr=-3.0, infl_snr=1.5, yfrac=0.05, max_snr=14)

        # if idx > 0:
        #     print(idx)
        #     data = bkg_subtracted_image[minx:maxx, miny:maxy]
        #     # out_array[:, idx] = corrected_image.reshape([-1])
        #     out_array = data.reshape([-1])
        #     out_array.astype('d').tofile(f)

        # corrected_image = observer.normalize_image(bkg_subtracted_image, max_val=255)
        # streak = Streak(corrected_image, remove_bkg='map', output_path=output_path)
        # streak.subtract_background()
        # corrected2 = streak.image

        # t0 = time.time()
        # enhanced_image = observer.gamma_correction(corrected_image, gamma=0.35, min_snr=3.0, infl_snr=7.0)
        # print('Levels Correction :: Time {0:4.4f}'.format(time.time() - t0))

        # if idx > 0:
        # t0 = time.time()
        # detected_peaks = observer.find_stars(corrected_image, nstars=10)
        # print('Peak Detection :: Time {0:4.4f}'.format(time.time() - t0))
        #
        # detx = detected_peaks[0][:, 0]
        # dety = detected_peaks[0][:, 1]
        #
        # # fid, ax = plt.subplots(1)
        # # ax.set_aspect('equal')
        # # ax.imshow(corrected_image, 'gray')
        #
        # for xx, yy in zip(detx, dety):
        #     # circ = Circle((xx, yy), 25)
        #     # ax.add_patch(circ)
        #     corrected_image = cv2.circle(corrected_image, (xx, yy), 25, (255, 0, 0), 3)

        # IPS_command = r'D:\Exo6Sim\bin\ips_win64_release.exe test.ofp' \
        #               r' -baseline_only -force_ct0 -thresh_spike 0.0' \
        #               r' -mcc_debug_level 99 -blur 1.5 -blur_det 1.5' \
        #               r' -write -ab -ch -em_multitarg' \
        #               r' -streak_axis_ratio 0.0 -force_cso -cso_detect'
        #
        # os.system(IPS_command)
        # observer.write_image(corrected_image, enhanced_dir)
        # observer.write_image(corrected_image, bg_sub_dir)

        observer.prev_image = observer.image
        prev_stars = cur_stars
