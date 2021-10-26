import os
import shutil
import subprocess
import datetime
import time

import threading
from queue import Queue

import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from classes.Conversions import dt2jd

import jdutil

from tkinter import *

from Camera import Camera

J2000 = 2451545.0
DTJ2000 = datetime.datetime(2000, 1, 1, 12)
EPSILON = 2.33E-10

image_dir = r'C:\Users\dtemple\AppData\Local\cygwin_ansvr'
launch_bash = 'C:/Users/dtemple/AppData/Local/cygwin_ansvr/bin/bash.exe --login -i '
output_path = r'C:\Users\dtemple\AppData\Local\cygwin_ansvr\tmp'
pixel_cmd = '--scale-units arcsecperpix --scale-low 22 --scale-high 24 --overwrite --downsample 4 --no-plots --crpix-center --no-remove-lines'


def pix2RD(fits_file='', x=0.0, y=0.0):
    f = fits.open(fits_file)
    mywcs = WCS(f[0].header)
    ra, dec = mywcs.all_pix2world([[x, y]], 0)[0]
    return ra, dec


def setup_camera(exposure_time):
    # os.system('D:/digiCamControl/CameraControl.exe')
    remote_command = 'D:/digiCamControl/CameraControlRemoteCmd.exe /c '
    shutter_command = remote_command + 'set shutterspeed "{}"'.format(exposure_time)
    os.system(shutter_command)
    return True


def capture_image(save_location=''):

    remote_command = 'D:/digiCamControl/CameraControlRemoteCmd.exe /c '

    jd_pre = dt2jd()

    t0 = time.time()
    # p = subprocess.Popen(full_command, stdout=subprocess.PIPE, universal_newlines=True, shell=False, close_fds=False)
    # (output, err) = p.communicate()
    remote_filename = os.path.join(save_location, str(jd_pre))
    remote_command_details = 'capture ' + remote_filename
    full_command = remote_command + ' ' + remote_command_details

    print('Beginning Exposure')
    os.system(full_command)
    print('Time to take exposure :: {0:4.4f}'.format(time.time() - t0))
    # if 'No connected device was found' in output:
    #     print(output)
    #     filename = 'test.txt'
    #     with open(filename, 'w') as f:
    #         f.write(filename)

    jd_post = dt2jd()
    jd_mid = (jd_post + jd_pre) / 2
    jdstr = str(jd_pre).ljust(20, '0')
    # remote_filename2 = remote_filename + '.jpg'
    max_wait = 5
    sleep_time = 0
    while not os.path.exist(remote_filename) and sleep_time < max_wait:
        time.sleep(0.01)
        sleep_time += 0.01

    if sleep_time >= max_wait:
        print('Image writing timeout')

    # os.rename(remote_filename2, remote_filename2.replace('image.jpg', jdstr + '.jpg'))

    with open(jdstr + '.txt', 'w') as f:
        f.writelines(str(jd_pre).ljust(20, '0') + '\n')
        f.writelines(str(jd_mid).ljust(20, '0') + '\n')
        f.writelines(str(jd_post).ljust(20, '0') + '\n')

    # This makes the wait possible
    # p_status = p.wait(1)
    # print(p.stdout.readline())

    # This will give you the output of the command being executed
    # if err is not None:
    #     print('Command output: ' + str(output))
    #     print('Command err: ' + str(err))

    return remote_filename


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
    return save_location


def subtract_median(image: Image, median_arr):
    im_arr = np.array(image, dtype=np.float32)
    return im_arr - median_arr


def gamma_correction(image0):
    min_snr = -3
    max_snr = 100
    infl_snr = 2.0
    yfrac = 0.1
    gamma = 0.45

    u = np.median(image0)
    s = np.sqrt(np.mean((image0 - u) ** 2))
    s = 1 if s == 0 else s

    pflat = image0.flatten()
    pflat = (pflat - u) / s
    za = pflat < min_snr
    zb = np.logical_and(pflat >= min_snr, pflat < infl_snr)  # values less than infl_snr
    zc = np.logical_and(pflat >= infl_snr, pflat <= max_snr)  # values greater than infl_snr
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


class CaptureEngine(object):

    time = ''

    shot_date = datetime.datetime.utcnow().strftime("%Y-%m-%d")

    scale = 0.6  # scale all fonts

    def __init__(self, save_dir):
        # createSaveFolder(os.path.join(save_dir, self.shot_date))
        threading.Thread.__init__(self)
        self.run()
        self.capturing = False

    def callback(self):
        self.root.quit()

    def run(self):
        self.root = Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.callback)

        self.clock_lt = Label(self.root, font=('arial', int(self.scale * 230), 'bold'), fg='red', bg='black')
        self.clock_lt.pack()

        self.date_iso = Label(self.root, font=('arial', int(self.scale * 75), 'bold'), fg='red', bg='black')
        self.date_iso.pack()

        self.root.mainloop()

    def capture_image(self, filename='image.jpeg', exposure_time=3):
        self.capturing = True
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
        new_file = jdstr + '.jpeg'
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

        self.capturing = False

        return new_file

    def tick(self):

        # dt = datetime.datetime.utcnow()
        wall_clock = time.strftime('%H:%M:%S')  # local
        # time_utc = time.strftime('%H:%M:%S', time.gmtime())  # utc
        # MJD
        # date_iso_txt = time.strftime('%Y-%m-%d', time.gmtime()) + "    %.10f" % jdutil.mjd_now()
        time2 = str(jdutil.mjd_now() + 2400000.5)

        # day, DOY, week
        # date_etc_txt = "%s   DOY: %s  Week: %s" % (time.strftime('%A'), time.strftime('%j'), time.strftime('%W'))

        # leap_secs = 37
        # gps_leap_secs = leap_secs - 19
        #
        # (gps_week, gps_s_w, gps_day, gps_s_day) = gpstime.gpsFromUTC(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, leapSecs=gps_leap_secs)
        # gps_hours = math.floor(gps_s_day / 3600.0)
        # gps_minutes = math.floor((gps_s_day - gps_hours * 3600.0) / 60.0)
        # gps_seconds = gps_s_day - gps_hours * 3600.0 - gps_minutes * 60.0
        # gps_txt = " GPS Time %02d:%02d:%02d \nGPS Week.Day %d.%d" % (gps_hours, gps_minutes, gps_seconds, gps_week, gps_day)
        # dt_tai = dt + datetime.timedelta(seconds=leap_secs)
        # tai_txt = " TAI %02d:%02d:%02d" % (dt_tai.hour, dt_tai.minute, dt_tai.second)

        if time2 != self.time:  # if time string has changed, update it
            self.time = time2
            self.clock_lt.config(text=wall_clock)
            # clock_utc.config(text=time_utc)
            self.date_iso.config(text=time2)
            # date_etc.config(text=date_etc_txt)
            # clock_gps.config(text=gps_txt)
            # clock_tai.config(text=tai_txt)

        if not self.capturing:
            self.capture_image(exposure_time=1/4000)

        self.clock_lt.after(20, self.tick)


def camera_capture_task(camera, save_dir):
    print('Capturing Image')
    shot_date = datetime.datetime.utcnow().strftime("%Y-%m-%d")  # This has been written to the while True loop.
    # save_location = createSaveFolder(os.path.join(save_dir, shot_date))
    shot_date
    time.sleep(1)
    # image_file = capture_image(save_location=save_location)
    # image_file = camera.capture_single_image(save_folder=save_location)
    shot_date
    return shot_date


def process_image_task(file, median_arr):
    print('Processing IMAGE FILE {}'.format(file))
    # image = Image.open(file).convert('L')  # converts to grayscale
    # image_arr = subtract_median(image, median_arr)  # subtracts median values for camera and lens
    # image_arr = gamma_correction(image_arr)
    #
    # temp_image = Image.fromarray(np.uint8(image_arr.clip(0, 255)))
    # temp_image.save(image_file, 'JPEG')


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
    save_dir = '../data'
    base_path = os.path.dirname(os.path.realpath(__file__)).replace('src', '')

    camera = Camera(image_type='.jpg', aperture=1.4, shutter_speed=0.5)
    camera.setup()
    # D5200 w/ 35mm
    # FOV :: 37.6 x 25.1 deg
    # pixel scale :: 22.6 arcsec/pixel

    # D5200 w/ 50mm
    # FOV ::
    # pixel scale :: 22.6 arcsec/pixel

    # while True:
    # timer = Ticker(save_dir)
    # timer.tick()
    # timer.root.mainloop()

    # Set up camera and parameters
    # initialize save directories
    # camera_setup = setup_camera(0.5)
    last_call = 0.0
    RATE_LIMIT = 1.5
    work = Queue()
    median_arr = np.load('median_array.npy')

    while True:

        next_job = camera_capture_task(camera, save_dir)

        if next_job:
            work.put(next_job)
        elif work.empty():
            break
        else:
            time.sleep(0.01)

        now = time.time()
        if now - last_call > RATE_LIMIT:
            last_call = now
            # print('Processing Image')
            process_image_task(work.get(), median_arr)
            # process_image_task()

        # # shot_date = datetime.datetime.utcnow().strftime("%Y-%m-%d")  # This has been written to the while True loop.
        # # save_location = createSaveFolder(os.path.join(save_dir, shot_date))
        # #
        # # # image_file = capture_image(save_location=save_location)
        # # image_file = camera.capture_single_image(save_folder=save_location)
        #
        # image = Image.open(file).convert('L')  # converts to grayscale
        # image_arr = subtract_median(image, median_arr)  # subtracts median values for camera and lens
        # image_arr = gamma_correction(image_arr)
        #
        # temp_image = Image.fromarray(np.uint8(image_arr.clip(0, 255)))
        # temp_image.save(image_file, 'JPEG')
        #
        # # image_file = 'C:/leo_tracking/src/test_image.jpg'
        # # output, err = plate_solve(image_file)
        #
        # # ff = read(image_file.replace('.jpg', '.new'))
        #
        # # streaks_detected = detect_streaks(image_file)
        # # >
