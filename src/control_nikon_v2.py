import os
import shutil
import subprocess
import datetime
import time

import cv2
import threading

import numpy as np

import imageio
from skimage import transform

from astropy.io import fits
from astropy.wcs import WCS
from classes.Conversions import dt2jd

# import jdutil

from tkinter import *

from classes.Camera import Camera
from PIL import Image

from classes.LeoProcessor import LEOProcessor

# import astroalign as aa
# from astride import Streak

image_dir = r'C:\Users\dtemple\AppData\Local\cygwin_ansvr'
launch_bash = 'C:/Users/dtemple/AppData/Local/cygwin_ansvr/bin/bash.exe --login -i '
output_path = r'C:\Users\dtemple\AppData\Local\cygwin_ansvr\tmp'
pixel_cmd = '--scale-units arcsecperpix --scale-low 22 --scale-high 24 --overwrite --downsample 4 --no-plots --crpix-center --no-remove-lines'


def capture_image(save_location=''):

    jd_pre = dt2jd()

    # t0 = time.time()
    # p = subprocess.Popen(full_command, stdout=subprocess.PIPE, universal_newlines=True, shell=False, close_fds=False)
    # (output, err) = p.communicate()
    remote_filename = os.path.join(save_location, str(jd_pre))
    remote_command = 'C:/digiCamControl/CameraControlRemoteCmd.exe /c '
    remote_command_details = 'capture ' + remote_filename
    full_command = remote_command + ' ' + remote_command_details

    print('Beginning Exposure')
    os.system(full_command)
    # print('Time to take exposure :: {0:4.4f}'.format(time.time() - t0))

    # jd_post = dt2jd()
    # jd_mid = (jd_post + jd_pre) / 2
    # jdstr = str(jd_pre).ljust(20, '0')
    # remote_filename2 = remote_filename + '.jpg'
    max_wait = 5
    sleep_time = 0
    while not os.path.exists(remote_filename) and sleep_time < max_wait:
        time.sleep(0.01)
        sleep_time += 0.01

    if sleep_time >= max_wait:
        print('Image writing timeout')

    # with open(jdstr + '.txt', 'w') as f:
    #     f.writelines(str(jd_pre).ljust(20, '0') + '\n')
    #     f.writelines(str(jd_mid).ljust(20, '0') + '\n')
    #     f.writelines(str(jd_post).ljust(20, '0') + '\n')

    return remote_filename


def plate_solve(image_path='C:/leo_tracking/src/test_image.jpg'):
    plate_command = launch_bash + '/bin/solve-field ' + pixel_cmd + ' ' + image_path
    p = subprocess.Popen(plate_command, stdout=subprocess.PIPE, universal_newlines=True, shell=False)
    (output, err) = p.communicate()
    return output, err


def camera_capture_task(camera, save_dir):
    shot_date = datetime.datetime.utcnow().strftime("%Y-%m-%d")  # This has been written to the while True loop.
    save_location = createSaveFolder(os.path.join(save_dir, shot_date))
    # image_file = capture_image(save_location=save_location)
    t0 = time.time()
    image_file = camera.capture_single_image(save_folder=save_location)
    print('Time to Capture Image :: {0:4.4f} seconds'.format(time.time() - t0))
    return image_file


def process_image_task(file, median_arr):
    print('Processing IMAGE FILE {}'.format(file))
    t0 = time.time()
    image = Image.open(file).convert('L')  # converts to grayscale
    image_arr = subtract_median(image, median_arr)  # subtracts median values for camera and lens

    if idx == 0:
        prev_aligned = np.zeros_like(image_arr)

    if idx != 0:
        prev_aligned = stackImagesKeypointMatching(image_arr, prev_arr)

    sub_arr = subtract_previous_image(image_arr, prev_aligned)
    otsu_arr = gamma_correction(sub_arr)

    temp_image = Image.fromarray(np.uint8(otsu_arr.clip(0, 255)))
    temp_image.save(image_file.replace('.jpg', '_c.jpg'), 'JPEG')
    print('Time to Process Image :: {0:4.4f} seconds\n'.format(time.time() - t0))


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

    """
    base_path = os.path.dirname(os.path.realpath(__file__)).replace('src', '')
    save_dir = os.path.join(base_path, 'data')

    camera = Camera(image_type='jpg', aperture=str(1.4), shutter_speed=str(1/2), iso=str(3200))
    # camera.setup()

    observer = LEOProcessor(base_path)
    observer.median_image = np.load('median_array.npy').astype(np.float32)

    shot_date = datetime.datetime.utcnow().strftime("%Y-%m-%d")  # This has been written to the while True loop.

    original_location = observer.create_save_folder(os.path.join(save_dir, shot_date, 'original'))
    processed_location = observer.create_save_folder(os.path.join(observer.save_dir, shot_date, 'processed'))

    idx = 0
    while True:

        t0 = time.time()
        camera.capture_single_image(save_folder=original_location)
        print('Image Num :: {0:5d}  Time to Capture :: {1:4.4f} seconds'.format(idx, time.time() - t0))

        # t0 = time.time()
        # image = Image.open(camera.image_name).convert('L')  # converts to grayscale
        # image_arr = subtract_median(image, median_arr)  # subtracts median values for camera and lens
        #
        # if idx == 0:
        #     prev_aligned = np.zeros_like(image_arr)
        #
        # if idx != 0:
        #     prev_aligned = stackImagesKeypointMatching(image_arr, prev_arr)
        #
        # sub_arr = subtract_previous_image(image_arr, prev_aligned)
        # otsu_arr = gamma_correction(sub_arr)
        #
        # temp_image = Image.fromarray(np.uint8(otsu_arr.clip(0, 255)))
        # pfile = camera.image_name.replace('original', 'processed')
        # temp_image.save(pfile, 'BMP')
        #
        # prev_arr = image_arr

        idx += 1

