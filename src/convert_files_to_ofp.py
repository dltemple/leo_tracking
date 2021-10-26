import os
import glob
import cv2
import numpy as np


IPS_command = r'D:\Exo6Sim\bin\ips_win64_release.exe test.ofp' \
                      r' -baseline_only -force_ct0 -thresh_spike 0.0' \
                      r' -mcc_debug_level 99 -blur 1.5 -blur_det 1.5' \
                      r' -write -ab -ch -em_multitarg' \
                      r' -streak_axis_ratio 0.0 -force_cso -cso_detect'

# os.system(IPS_command)

base_dir = r'D:\leo_tracking\data\2020-02-23\processed\sat2\c'
base_dir = base_dir.replace('\\', '/')

odir = r'D:\leo_tracking\data\2020-02-23\processed\sat2\d'
odir = odir.replace('\\', '/')

images = glob.glob(os.path.join(base_dir, '*' + '.jpg'))
images = images[1:]


minx, maxx = 2000-256, 2000+256
miny, maxy = 3556-256, 3556+256

out_array = np.zeros([512 * 512, len(images)])

f = open('test_512.ofp', 'wb')
for idx, image in enumerate(images):
    image_data = cv2.imread(image, -1)
    image_data = image_data[minx:maxx, miny:maxy]
    fname = image.replace('/c\\', '/d/')

    cv2.imwrite(fname, image_data)
    out_array[:, idx] = image_data.reshape([-1])

out_array.astype('d').tofile(f)
np.array(512).astype('int').tofile(f)
np.array(512).astype('int').tofile(f)
np.array(idx + 1).astype('int').tofile(f)

f.close()

exit()
