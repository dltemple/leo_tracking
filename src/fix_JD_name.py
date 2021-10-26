import os
import glob

base_dir = 'D:\\leo_tracking\\data\\2020-03-06\\original'
# images = glob.glob('c:/leo_tracking/data/titanIV_grey/*.jpeg')
jpgs = glob.glob(os.path.join(base_dir, '*.jpg'))
raws = glob.glob(os.path.join(base_dir, '*.raw'))
nefs = glob.glob(os.path.join(base_dir, '*.nef'))

# for image in jpgs:
#
#     ps = image.split('\\')[-1]
#     ps = ps.split('.jpg')[0]
#     new_jd = ps.ljust(20, '0')
#     nfile = new_jd + '.jpg'
#     os.rename(os.path.join(base_dir, image), os.path.join(base_dir, nfile))

ext = '.nef'
for image0 in nefs:
    image = image0.replace(ext, '')
    ps = image.split('\\')[-1]
    ps = ps.split(ext)[0]
    new_jd = ps.ljust(20, '0')
    nfile = new_jd + ext
    os.rename(os.path.join(base_dir, image0), os.path.join(base_dir, nfile))