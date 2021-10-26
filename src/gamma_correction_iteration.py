import os
import glob
import time
import numpy as np

from PIL import Image
from skimage import transform


def gamma_correction(image, gamma=0.45, infl_snr=1.0, max_snr=100, min_snr=-3, yfrac=0.1, output='8'):
    # min_snr = -3
    # max_snr = 100
    # infl_snr = 2.0
    # yfrac = 0.1
    # gamma = 0.45
    # if self.dark_vals is not None:
    #     # mask dark values for noise analysis
    #     masked_image = image[self.dark_vals] = 0.0
    # else:
    masked_image = image[:, 1:]

    u = np.median(masked_image)
    s = np.sqrt(np.mean((masked_image - u) ** 2))
    s = 1 if s == 0 else s

    image = (image - u) / s

    za = image < min_snr
    zb = np.logical_and(image >= min_snr, image < infl_snr)
    zc = np.logical_and(image >= infl_snr, image <= max_snr)
    zd = image > max_snr

    image[za] = 0.
    image[zb] = yfrac * ((image[zb] - min_snr) / (infl_snr - min_snr)) ** (1 / gamma)
    image[zc] = yfrac + (1 - yfrac) * ((image[zc] - infl_snr) / (max_snr - infl_snr)) ** gamma
    image[zd] = 1.

    # if self.dark_vals is not None:
    #     image[self.dark_vals] = 0.0
    #     image[:, 0] = 0.0
    #     image[0, :] = 0.0
    image[:, 0] = 0.0

    if output == '16':
        return normalize_image(image, 65535)
    else:
        return normalize_image(image, 255)


def normalize_image(image, max_val=255):
    if max_val == 65535:
        return (max_val * ((image - image.min()) / image.ptp())).astype(np.uint16)
    elif max_val == 255:
        return (max_val * ((image - image.min()) / image.ptp())).astype(np.uint8)


def create_save_folder(save_location):
    try:
        os.makedirs(save_location)
    except Exception:
        pass
    os.chdir(save_location)
    return save_location


if __name__ == '__main__':
    ext = '.jpg'

    processed_dir = 'C:/leo_tracking/data/2020-02-23/processed/sat2/b/'
    # images = glob.glob(os.path.join(base_dir, '*' + ext))
    # processed_images = glob.glob(os.path.join(processed_dir, '*' + ext))

    # createSaveFolder(processed_dir)
    # createSaveFolder(os.path.join(processed_dir, 'a'))
    # createSaveFolder(os.path.join(processed_dir, 'b'))
    # createSaveFolder(os.path.join(processed_dir, 'c'))
    out_dir = create_save_folder(os.path.join(processed_dir, 'g'))

    base_image = os.path.join(processed_dir, '2458902.545947215500_b.jpg')
    image_arr = np.array(Image.open(base_image))

    image_arr = transform.rescale(image_arr, scale=0.25)

    gammas = np.linspace(0.05, 1.0, 19).tolist()
    infls = np.linspace(1, 8, 19).tolist()
    min_snrs = np.linspace(-5, 2, 8).tolist()
    y_fracs = np.linspace(0.05, 1.0, 19).tolist()

    min_snr = 3
    y_frac = 0.05
    infl = 6.0
    gamma = 0.35

    for gamma in gammas:
        # for infl in infls:
            # for min_snr in min_snrs:
                # for y_frac in y_fracs:
                    # t0 = time.time()
                    out_arr = gamma_correction(image_arr, gamma=gamma, infl_snr=infl, max_snr=100, min_snr=min_snr, yfrac=y_frac)
                    # print('Levels Correction :: Time {0:4.4f}'.format(time.time() - t0))
                    print('Gamma : {0:2.2f}    Infl : {1:2.2f}    MSNR : {2:2.2f}    FRAC : {3:2.2F}'.format(gamma, infl, min_snr, y_frac))

                    ostr = 'G{}_I{}_S{}_F{}'.format(gamma, infl, min_snr, y_frac)

                    temp_image = Image.fromarray(np.uint8(out_arr.clip(0, 255)))
                    fname = os.path.join(out_dir, ostr)
                    temp_image.save(fname + ext, 'JPEG')