import os
import shutil
import subprocess
import datetime
import time
import glob
import math

import numpy as np
from time import sleep

from astropy.io import fits
from astropy.wcs import WCS

from FFfits import read
from FFStruct import FFStruct

J2000 = 2451545.0
DTJ2000 = datetime.datetime(2000, 1, 1, 12)
EPSILON = 2.33E-10

# plate solver api
# http://127.0.0.1:8080/api

# import astropy as ap
# import astrometry_azel as ael
# from astropy.io import fits
#
# try:
#     import astrometry_azel.plots as aep
#     from matplotlib.pyplot import show
# except ImportError:
#     show = None
from matplotlib import pyplot as plt

image_dir = r'C:\Users\dtemple\AppData\Local\cygwin_ansvr'

launch_bash = 'C:/Users/dtemple/AppData/Local/cygwin_ansvr/bin/bash.exe --login -i '

output_path = r'C:\Users\dtemple\AppData\Local\cygwin_ansvr\tmp'

pixel_cmd = '--scale-units arcsecperpix --scale-low 15 --scale-high 16 --pixel-error 2 --overwrite --downsample 4 --no-plots --crpix-center --no-remove-lines'


def pix2RD(fits_file='', x=0.0, y=0.0):
    f = fits.open(fits_file)
    mywcs = WCS(f[0].header)
    ra, dec = mywcs.all_pix2world([[x, y]], 1)[0]

    # ax = plt.subplot(projection=mywcs, label='overlays')
    # ax.coords.grid(True, color='white', ls='solid')
    # ax.coords[0].set_axislabel('Galactic Longitude')
    # ax.coords[1].set_axislabel('Galactic Latitude')
    # ax.imshow(f[0].data, origin='lower', cmap=plt.cm.viridis)
    # # ax.xlabel('RA')
    # # ax.ylabel('Dec')
    # overlay = ax.get_coords_overlay('fk5')
    # overlay.grid(color='white', ls='dotted')
    # overlay[0].set_axislabel('Right Ascension (J2000)')
    # overlay[1].set_axislabel('Declination (J2000)')
    # plt.scatter(x, y, c='y', marker='*')
    # plt.pause(0.01)

    return ra, dec


def capture_image(filename='image.jpg', exposure_time=3):
    exp_time = str(exposure_time)
    camera_command = 'D:/digiCamControl/CameraControlCmd.exe'
    camera_command_details = '/filename ' + './' + filename + ' /capturenoaf /iso 1600 /shutter ' + exp_time + ' /aperture 1.4'
    # print('camera details = ', camera_command_details)
    full_command = camera_command + ' ' + camera_command_details

    jd_pre = dt2jd()

    t0 = time.time()
    p = subprocess.Popen(full_command, stdout=subprocess.PIPE, universal_newlines=True, shell=False)
    (output, err) = p.communicate()

    print('Time to take exposure :: {0:4.4f}'.format(time.time() - t0))
    if 'No connected device was found' in output:
        print(output)
        filename = 'test.txt'
        with open(filename, 'w') as f:
            f.write(filename)

    jd_post = dt2jd()
    jd_mid = (jd_post + jd_pre) / 2

    jdstr = str(jd_mid).ljust(20, '0')
    new_file = jdstr + '.jpg'
    shutil.move(filename, new_file)

    with open(jdstr + '.txt', 'w') as f:
        f.writelines(str(jd_pre).ljust(20, '0') + '\n')
        f.writelines(str(jd_mid).ljust(20, '0') + '\n')
        f.writelines(str(jd_post).ljust(20, '0') + '\n')

    # This makes the wait possible
    # p_status = p.wait(1)
    # print(p.stdout.readline())

    # This will give you the output of the command being executed
    if err is not None:
        print('Command output: ' + str(output))
        print('Command err: ' + str(err))

    return new_file


def plate_solve(image_path='C:/leo_tracking/src/test_image.jpg'):
    plate_command = launch_bash + '/bin/solve-field ' + pixel_cmd + ' ' + image_path
    p = subprocess.Popen(plate_command, stdout=subprocess.PIPE, universal_newlines=True, shell=False)
    (output, err) = p.communicate()
    return output, err


def detect_streaks(image_path=''):
    return True


def createSaveFolder(save_location):
    try:
        os.makedirs(save_location)
    except Exception:
        pass
    os.chdir(save_location)


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

    base_dir = '../data/2020-02-23/sat2/'
    # images = glob.glob('c:/leo_tracking/data/titanIV_grey/*.jpeg')
    images = glob.glob(os.path.join(base_dir, '*.jpg'))
    times = glob.glob(os.path.join(base_dir, '*.txt'))

    times = [f for f in times if 'centroids' not in f]
    # D5200 w/ 35mm
    # FOV :: 37.6 x 25.1 deg
    # pixel scale :: 22.6 arcsec/pixel

    # D5200 w/ 50mm
    # FOV :: 26.1 x 17.4 deg
    # pixel scale :: 15.7 arcsec/pixel

    # TITAN IV
    # pixel_centroids = [[2252.5, 461],
    #                    [1690.5, 992],
    #                    [1143.5, 1526.5],
    #                    [622.5, 2011.5],
    #                    [127.5, 2492.5]]

    # SAT1 02-20-21
    # pixel_centroids = [[2245.97, 350.82],
    #                    [2178.70, 730.51],
    #                    [2122.17, 1096.48],
    #                    [2054.87, 1463.52],
    #                    [1991.27, 1821.89],
    #                    [1935.15, 2151.85],
    #                    [1882.53, 2484.97],
    #                    [1826.28, 2821.57],
    #                    [1775.92, 3132.10],
    #                    [1713.20, 3505.88],
    #                    [1667.50, 3784.03],
    #                    ]

    # SAT2 02-21-20
    pixel_centroids = np.loadtxt(base_dir + 'pixel_centroids.txt', delimiter=',')

    # for idx, image in enumerate(images):
    #     print('Plate Solving Image :: {0}'.format(image))
    #     output, err = plate_solve(image)
    #
    # exit()

    ra_dec_centroids = list()
    for idx, img_time in enumerate(zip(images[:], times[:])):

        image = img_time[0]
        time = img_time[1]

        with open(time, 'r') as f:
            all_times = f.readlines()

        jd_start, jd_mid, jd_stop = map(float, all_times)

        # output, err = plate_solve(image)

        # ff = read(image.replace('.jpeg', '.new'))
        px = pixel_centroids[idx][1]
        py = pixel_centroids[idx][2]

        try:
            ra, dec = pix2RD(image.replace('.jpg', '.new'), px, py)
            print('PX {0:4.4f}  PY {1:4.4f} RA :: {2:4.6f}  DEC :: {3:4.6f}'.format(px, py, ra, dec))
            # streaks_detected = detect_streaks(image)
            ra_dec_centroids.append(str([jd_start, ra, dec]).replace('[', '').replace(']', ''))
        except Exception:
            print('Plate Solving Failed')
            ra_dec_centroids.append(str([9999.9999, -1, -1]).replace('[', '').replace(']', ''))
            continue

    with open(os.path.join(base_dir, 'ra_dec_centroids.txt'), 'w') as f:
        f.writelines('\n'.join(ra_dec_centroids))

    ra_dec_centroids