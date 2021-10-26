import datetime
# import rawpy
import glob
import os
import subprocess

import cv2
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from photutils import find_peaks


class LEOProcessor(object):
    def __init__(self, base_path):
        self.base_path = base_path
        self.save_dir = os.path.join(self.base_path, 'data')

        self.save_location = ''

        self.image_file = ''
        self.image_dir = ''

        self.image_original = None
        self.image = None
        self.prev_image = None
        self.median_image = None
        self.dark_vals = None

        self.pmedian = None
        self.pstd = None

        self.gif_framerate = 0.25

        self.original_image_size = (6000, 4000)
        self.reduced_image_size = (3000, 2000)

        pass

    def find_stars(self, image: np.ndarray, nstars: int = 500, sigma: float = 3.0, box_size: int = 11) -> np.ndarray:
        if self.reduced_image_size != self.original_image_size:
            image = cv2.resize(image, self.reduced_image_size, cv2.INTER_AREA)

        pmean, pmedian, pstd = sigma_clipped_stats(image, sigma=sigma)
        threshold = pmedian + (5.0 * pstd)
        data = find_peaks(image, threshold, box_size=box_size, npeaks=500)
        points = np.array([data['x_peak'], data['y_peak'], data['peak_value']])
        sarr = points[-1, :].argsort()
        points = points[:, sarr[::-1]].T
        return points[:nstars], pmedian, pstd

    def stack_images_keypoints(self, prev_points=None):

        cur_points, pmedian, pstd = self.find_stars(self.image)

        if prev_points is None:
            prev_points, pmedian, pstd = self.find_stars(self.prev_image)

        # create BFMatcher object
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Find matches and sort them in the order of their distance
        matches = matcher.match(cur_points[:, :2].astype(np.float32),
                                prev_points[:, :2].astype(np.float32))

        vmatch, vdist = list(), list()
        for midx, match in enumerate(matches):
            img1_idx, img2_idx = match.queryIdx, match.trainIdx

            a = cur_points[img1_idx, :2].astype(np.int32)
            b = prev_points[img2_idx, :2].astype(np.int32)

            dist = np.linalg.norm(a - b)
            if dist < 10:
                vmatch.append(match)
                vdist.append(dist)

        sorted_y_idx_list = sorted(range(len(vdist)), key=lambda x: vdist[x])
        good = [vmatch[i] for i in sorted_y_idx_list][0:int(len(vmatch))]

        good = sorted(good, key=lambda x: x.distance)
        good = good[0:int(len(good))]

        # print("    > after filtration {} matches left\n".format(len(good)))

        # getting source & destination points
        src_pts = np.float32([cur_points[m.queryIdx, :2] for m in good]).reshape(-1, 2)
        dst_pts = np.float32([prev_points[m.trainIdx, :2] for m in good]).reshape(-1, 2)

        # Estimate perspective transformation
        M, mask = cv2.estimateAffine2D(dst_pts.astype(np.float32), src_pts.astype(np.float32), method=cv2.RANSAC, confidence=0.98)
        print('{0:1.2e} {1:1.2e} {2:1.2e}'.format(M[0, 0], M[0, 1], M[0, 2]))
        print('{0:1.2e} {1:1.2e} {2:1.2e}'.format(M[1, 0], M[1, 1], M[1, 2]))

        # M[0, 0] = 1.
        # M[1, 1] = 1.
        # M[0, 1] = 0.0
        # M[1, 0] = 0.0

        w, h = self.image.shape
        prev_image_align = cv2.warpAffine(self.prev_image, M, (h, w))
        # self.dark_vals = prev_image_align <= 2 * pstd
        # self.dark_vals = prev_image_align == 0.0
        self.dark_vals = None
        # prev_image_align[dark_vals] = pmedian

        return prev_image_align, cur_points, pmedian, pstd

    def clip_values_uint8(self, image):
        image[image < 0] = 0
        image[image > 255] = 255
        return image

    def clip_values_uint16(self, image):
        image[image < 0] = 0
        image[image > 65535] = 65535
        return image

    def subtract_image(self, prev_image):
        sub_image = self.image - prev_image
        # some values will be less than 0. need to adjust (but not to zero)
        # sub_image = self.shift_vals(sub_image)
        # self.clip_values_uint8(sub_image)
        return sub_image

    def subtract_median(self):
        self.image = self.image_original - self.median_image
        # self.image = self.shift_vals(self.image)
        # self.image = self.clip_values_uint8(self.image)
        # self.image = self.normalize_image(sub_image, 255)

    def shift_vals(self, image):
        mv = np.min(image)
        if np.min(image) < 0:
            return image - mv
        else:
            return image

    def normalize_image(self, image, max_val):
        if max_val == 65535:
            return (max_val * ((image - image.min()) / image.ptp())).astype(np.uint16)
        elif max_val == 255:
            return (max_val * ((image - image.min()) / image.ptp())).astype(np.uint8)

    def gamma_correction(self, image, gamma=0.45, infl_snr=1.0, max_snr=100, min_snr=-3, yfrac=0.1, output='8'):
        # min_snr = -3
        # max_snr = 100
        # infl_snr = 2.0
        # yfrac = 0.1
        # gamma = 0.45
        if self.dark_vals is not None:
            # mask dark values for noise analysis
            masked_image = image[self.dark_vals] = self.pmedian
        else:
            masked_image = image

        u = np.mean(masked_image)
        s = np.sqrt(np.mean((masked_image - u) ** 2))
        s = 1 if s == 0 else s

        # plt.figure()
        # plt.imshow(masked_image, 'gray')
        # plt.pause(0.01)
        # for _ in range(3):
        #     u = np.mean(masked_image)
        #     s = np.sqrt(np.mean((masked_image - u) ** 2))
        #     s = 1 if s == 0 else s
        #
        #     print('Median :: {0:2.2f}    STD :: {1:2.2f}'.format(u, s))
        #
        #     masked_image[(masked_image - u) > 3.0 * s] = 0.0
        #     plt.figure()
        #     plt.imshow(masked_image, 'gray')
        #     plt.pause(0.01)

        image = (image - u) / s

        za = image < min_snr
        zb = np.logical_and(image >= min_snr, image < infl_snr)
        zc = np.logical_and(image >= infl_snr, image <= max_snr)
        zd = image > max_snr

        image[za] = 0.
        image[zb] = yfrac * ((image[zb] - min_snr) / (infl_snr - min_snr)) ** (1 / gamma)
        image[zc] = yfrac + (1 - yfrac) * ((image[zc] - infl_snr) / (max_snr - infl_snr)) ** gamma
        image[zd] = 1.

        if self.dark_vals is not None:
            image[self.dark_vals] = 0.0
            image[:, 0] = 0.0
            image[0, :] = 0.0

        # if output == '16':
        #     return self.normalize_image(image, 65535)
        # else:
        #     return self.normalize_image(image, 255)

        if output == '16':
            return image * 65535
        else:
            return image * 255

    def create_gif(self, filenames, output_file=None):
        gif_images = list()
        for filename in filenames:
            image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            # gif_images.append(transform.rescale(image, scale=0.25))
            gif_images.append(image)
        if output_file is None:
            output_file = 'Gif-%s.gif' % datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S')

        imageio.mimsave(output_file, gif_images, duration=self.gif_framerate)

    @property
    def fits_ext(self):
        return ['*.axy', '*.corr', '*.match', '*.solved', '*.new', '*.xyls', '*.rdls']

    def remove_files(self, ext='*.axy'):
        files = glob.glob(os.path.join(self.image_dir, ext))
        for file in files:
            os.remove(file)

    def pix2RD(self, fits_file='', x=0.0, y=0.0):
        f = fits.open(fits_file)
        mywcs = WCS(f[0].header)
        ra, dec = mywcs.all_pix2world([[x, y]], 0)[0]
        return ra, dec, mywcs

    def plate_solve(self, image_path='C:/leo_tracking/src/test_image.jpg', ra=None, dec=None):
        launch_bash = 'C:/Users/dtemple/AppData/Local/cygwin_ansvr/bin/bash.exe --login -i '
        pixel_cmd = '--scale-units arcsecperpix --scale-low 15 --scale-high 16 --pixel-error 2 --overwrite --downsample 8 --no-plots --crpix-center --no-remove-lines'

        if ra:
            ra_command = ' -ra {0:4.6f}'.format(ra)
        else:
            ra_command = ''

        if dec:
            dec_command = ' -dec {0:4.6f}'.format(dec)
        else:
            dec_command = ''

        plate_command = launch_bash + '/bin/solve-field ' + pixel_cmd + ' ' + image_path + ra_command + dec_command
        p = subprocess.Popen(plate_command, stdout=subprocess.PIPE, universal_newlines=True, shell=False)
        (output, err) = p.communicate()

        for ext in self.fits_ext:
            self.remove_files(ext)

        fits_file = image_path.replace('.jpg', '.wcs')
        return output, err, fits_file

    def create_save_folder(self, save_location):
        try:
            os.makedirs(save_location)
        except Exception:
            pass
        os.chdir(save_location)
        return save_location

    def setup_camera(self, exposure_time):
        # os.system('D:/digiCamControl/CameraControl.exe')
        remote_command = 'C:/digiCamControl/CameraControlRemoteCmd.exe /c '
        shutter_command = remote_command + 'set shutterspeed "{}"'.format(exposure_time)
        os.system(shutter_command)
        return True

    @staticmethod
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    def read_image(self, image_file):
        self.image_file = image_file
        self.fname_in = image_file.split(os.sep)[-1]
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        # with rawpy.imread(image_file) as raw:
        #     raw_image = raw.postprocess(output_bps=8).copy()
        #     # raw_image = raw.raw_image.copy()
        #
        # raw_image = self.rgb2gray(raw_image)
        #
        # d1 = (4020 - 4000) / 2
        # d2 = (6036 - 6000) / 2
        # z1 = list(range(int(d1), int(4020 - d1)))
        # z2 = list(range(int(d2), int(6036 - d2)))
        #
        # raw_image = raw_image[z1[0]:z1[-1]+1, z2[0]:z2[-1]+1]
        # image = cv2.resize(raw_image, (6000, 4000), interpolation=cv2.INTER_AREA)
        #
        # del raw_image
        self.image_original = np.array(image, dtype=np.float32)

    def write_image(self, image_arr, output_dir, file_ext='.jpg'):
        fname = self.fname_in[:-4]
        ps = fname.split(os.sep)
        imname = ps[-1].ljust(20, '0')
        fout = os.path.join(output_dir, imname + file_ext)
        cv2.imwrite(fout, image_arr.astype(np.uint8))

    def get_output_filename(self, label='', ext='.jpg'):
        pass

    def run_ips(self):
        IPS_command = r'C:\leo_tracking\src\ips_win64_release.exe test.ofp' \
                      r' thresh_spike 1.0 -thresh_pixel 4.5 -thresh_mf' \
                      r'12.0 -mcc_debug_level 0 -blur 3.0 -blur_det 3.0' \
                      r'-write  -streak_axis_ratio 0.0'

        os.system(IPS_command)
