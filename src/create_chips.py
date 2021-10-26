import glob
import os

from PIL import Image

from classes.LeoProcessor import LEOProcessor


# import jdutil
# import gpstime
# import threading
# from queue import Queue
# import imageio
# from skimage import transform


def crop(infile, height, width):
    im = Image.open(infile)
    imgwidth, imgheight = im.size
    for i in range(imgheight // height):
        for j in range(imgwidth // width):
            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
            yield im.crop(box)


ext = '.jpg'

base_path = os.path.dirname(os.path.realpath(__file__)).replace('src', '')
save_dir = os.path.join(base_path, 'data')

image_dir = os.path.join(save_dir, '2020-03-26/original/set2/')
processed_dir = image_dir.replace('original', 'processed')

observer = LEOProcessor(base_path)

images = glob.glob(os.path.join(image_dir, '*' + ext))
# processed_images = glob.glob(os.path.join(processed_dir, '*' + ext))

observer.create_save_folder(processed_dir)
# createSaveFolder(os.path.join(processed_dir, 'a'))
# createSaveFolder(os.path.join(processed_dir, 'b'))
enhanced_dir = observer.create_save_folder(os.path.join(processed_dir, 'c'))
bg_sub_dir = observer.create_save_folder(os.path.join(processed_dir, 'e'))

enhanced_images = glob.glob(os.path.join(enhanced_dir, '*' + '.jpg'))

height, width = 128, 128

for idx, image in enumerate(images):
    for k, piece in enumerate(crop(image, height, width)):
        img = Image.new('L', (height, width), 0)
        img.paste(piece)
        path = os.path.join('C:/leo_tracking/results/chips/set1', "IMG-%s.jpg" % k)
        img.save(path)
