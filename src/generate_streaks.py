import copy
import cv2
import glob
import matplotlib.pyplot as plt
import os
import random
import shutil
from matplotlib import path
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
from scipy.special import erf

from detection import *

import numpy as np
import math

from fkdat import fake_data

SP = FPA()

TFRAMEX, TFRAMEY = 4020, 6036
FRAMEX, FRAMEY = 4000, 6000

# TFRAMEX, TFRAMEY = 402, 604
# FRAMEX, FRAMEY = 400, 600

d1 = (TFRAMEX - FRAMEX) // 2
d2 = (TFRAMEY - FRAMEY) // 2

zx = list(range(int(d1), int(TFRAMEX - d1 + 1)))
zy = list(range(int(d2), int(TFRAMEY - d2 + 1)))

bg_low, bg_high = -0.1, 0.1
std_low, std_high = 3.0, 5.0

bg_mean = np.random.uniform(bg_low, bg_high, 1)
bg_std = np.random.uniform(std_low, std_high, 1)

fpC = np.random.normal(loc=bg_mean, scale=bg_std, size=(TFRAMEX, TFRAMEY))
fpC = fpC[zx[0]:zx[-1], zy[0]:zy[-1]]

D1 = fpC.shape[0]
D2 = fpC.shape[1]

lengths = list(range(5, 30, 1))
snrs = list(range(2, 15))
angles = list(range(180))

bg_low, bg_high = -0.1, 0.1
std_low, std_high = 3.0, 5.0

t = np.linspace(-1, 1, 41)

# Create images and then subdivide into ROI that are 128 x 128
for image in range(5000):

    fpC, kernel_sums = fake_data(TFRAMEX, TFRAMEY)

    bg_mean = np.mean(fpC)
    bg_std = np.std(fpC)
    # focal plane noise parameters
    # bg_mean = np.random.uniform(bg_low, bg_high, 1)
    # bg_std = np.random.uniform(std_low, std_high, 1)
    # fpC = np.random.normal(loc=bg_mean, scale=bg_std, size=(TFRAMEX, TFRAMEY))
    fpC = fpC[zx[0]:zx[-1], zy[0]:zy[-1]]

    D1, D2 = fpC.shape

    n_streaks = random.randint(1, 2)

    streak_info = list()
    for sidx in range(4):
        # create streaks with varying lengths, angles, snrs
        # inject bright point sources randomly that should be ignored
        temp_length = random.choice(lengths)
        temp_snr = random.choice(snrs)
        temp_angle = random.choice(angles)

        mask_dim = math.ceil(temp_length) * 2

        ang = math.radians(temp_angle)

        s12 = np.array([(temp_length * math.cos(ang) / 2 * t + mask_dim / 2),
                        (temp_length * math.sin(ang) / 2 * t + mask_dim / 2)]).T

        tm = target_streak_matrix_01(mask_dim, mask_dim, SP.blur, s12)
        streak_flux = np.random.randint(5000, 68000)

        tm *= streak_flux

        bdim = mask_dim // 10
        tm = cv2.blur(tm, (bdim, bdim))

        sdist = tm.shape[0] // 2 + 1

        # streak can only be +- mask from edge of frame
        min_x, max_x = sdist, FRAMEX - sdist
        min_y, max_y = sdist, FRAMEY - sdist

        # Streak centroids
        x_centroid = random.randint(min_x, max_x)
        y_centroid = random.randint(min_y, max_y)

        fpidx = list(range(-sdist+1, sdist))
        fpx = [_ + x_centroid for _ in fpidx]
        fpy = [_ + y_centroid for _ in fpidx]
        fpC[fpx[0]:fpx[-1], fpy[0]:fpy[-1]] += tm

    # plt.imshow(fpC, 'gray')
    # plt.pause(0.01)

    # tmp = fpC  # for drawing a rectangle
    # x_step = 600
    # y_step = 400
    # (w_height, w_width) = (400, 600)  # window size
    # for y in range(0, fpC.shape[1] - w_width, x_step):
    #     for x in range(0, fpC.shape[0] - w_height, y_step):
    #         window = fpC[x:x + w_height, y:y + w_width]
    #
    #         cv2.rectangle(tmp, (x, y), (x + w_width, y + w_height), (255, 0, 0), 2)  # draw rectangle on image
    #         plt.imshow(np.array(tmp).astype('uint8'))
    #
    # plt.show()

for r in r_list:

    for tht1 in range(16):
        pmasks, glist, cso_com_list = [], [], []
        n_plumes = plume_list[iters]

        for pn in range(n_plumes):

            blur_add = (random.random() * 0.5) - 0.2
            erf_blur = erf(ob.eff_blur + blur_add)

            if pn == 0:
                # pobject = np.zeros_like(pdata)
                pobject = copy.copy(pdata)

                cur_cso_mask = np.zeros_like(pdata)
                xi = random.randint(-1, 1)
                yi = random.randint(-1, 1)

                rso_com = ndimage.measurements.center_of_mass(pobject)

            r_space = np.linspace(0, r, 25)

            theta_range = np.linspace(0, 2 * np.pi, 25)
            x_vertices0 = r_space * np.cos(th2)
            y_vertices0 = r_space * np.sin(th2)

            x_vertices1 = x_vertices0 + np.cos(w_random)
            y_vertices1 = y_vertices0 + np.sin(w_random)

            x_vertices = np.concatenate([x_vertices0, x_vertices1], axis=0)
            y_vertices = np.concatenate([y_vertices0, y_vertices1], axis=0)

            # r1 = r
            # r2 = r1 + (2 * r1*0.2) * random.random() - r1*0.2
            # x_pos1 = r1 * np.cos(th1)
            # y_pos1 = r1 * np.sin(th1)
            #
            # x_pos2 = r2 * np.cos(th2)
            # y_pos2 = r2 * np.sin(th2)

            move_x = random.randint(-5, 5)
            move_y = random.randint(-5, 5)

            # move_x = 0
            # move_y = 0

            centx = move_x + rso_com[0]
            centy = move_y + rso_com[1]

            x_vertices += centx
            y_vertices += centy

            # xloc1 = centx + x_pos1  # xloc of RSO
            # yloc1 = centy + y_pos1  # yloc of RSO
            #
            # xloc2 = centx + x_pos2  # xloc of RSO
            # yloc2 = centy + y_pos2  # yloc of RSO

            # total_area = np.pi * r**2 * np.abs(th2-th1) / (2*np.pi)
            # arc_area = 0.5 * r ** 2 * (np.abs(th2-th1) - np.sin(np.abs(th2-th1)))
            # area_of_interest = total_area - arc_area

            total_intensity = intensity * pmax * np.sqrt(n_plumes)

            # import matplotlib.pyplot as plt

            first = 0
            size = 1
            xv, yv = np.meshgrid(np.linspace(0, 31, 32), np.linspace(0, 31, 32))
            verts = list(zip(list(x_vertices), list(y_vertices)))

            p = path.Path(verts)
            flags = p.contains_points(np.hstack((xv.flatten()[:, np.newaxis], yv.flatten()[:, np.newaxis])))
            grid = np.zeros((32, 32), dtype='bool')

            grid[((xv.flatten() - first) / size).astype('int'), ((yv.flatten() - first) / size).astype('int')] = flags

            # xi, yi = np.random.randint(0, 31, 32), np.random.randint(0, 31, 32)
            # vflag = grid[((xi - first) / size).astype('int'), ((yi - first) / size).astype('int')]
            # plt.imshow(grid.T, origin='lower', interpolation='nearest', cmap='binary')
            # plt.scatter(((xi - first) / size).astype('int'), ((yi - first) / size).astype('int'), c=vflag, cmap='Greens', s=90)
            # plt.show()
            grid.resize()
            trues = np.sum(grid)
            avg_int = total_intensity / trues
            int_mask = np.zeros_like(cur_cso_mask)
            remaining_int = total_intensity
            remaining_true = trues
            while remaining_true:
                for ri in range(grid.shape[0]):
                    for ci in range(grid.shape[1]):
                        if grid[ri, ci]:
                            # max intensity is 2 times average
                            # min intensity is 0.1 times average
                            int_sample = (1 * random.random() + 0.2) * avg_int

                            # int_sample = (1 / remaining_true) * random.random() * remaining_int
                            int_mask[ri, ci] = int_sample
                            remaining_int -= int_sample
                            remaining_true -= 1

            if pn > 0:
                r1 = False
                r2 = False
                if random.random() > 0.5 or r1 is True:
                    int_mask = np.fliplr(int_mask)
                    grid = np.fliplr(grid)
                    r1 = True
                if random.random() > 0.5 or r2 is True:
                    int_mask = np.flipud(int_mask)
                    grid = np.flipud(grid)
                    r2 = False
                # if random.random() > 0.9 and (r1 is False or r2 is False):
                #     int_mask2 = np.flipud(int_mask)
                #     grid2 = np.flipud(grid)
                #     int_mask = int_mask + int_mask2
                #     grid = grid + grid2

            # plt.imshow(int_mask)

            int_mask = gaussian_filter(int_mask, erf_blur, truncate=5)
            cso_com = ndimage.measurements.center_of_mass(int_mask)

            pmasks.append(int_mask)
            cso_com_list.append(cso_com)
            glist.append(grid)

        rand_bg = np.random.normal(loc=bg_mean, scale=bg_std, size=np.shape(pdata))

        pdata_cso = rand_bg
        for pmask in pmasks:
            pdata_cso += pmask

        # cso_com = ndimage.measurements.center_of_mass(int_mask)

        pdata_cso = (65535 * ((pdata_cso - pdata_cso.min()) / pdata_cso.ptp())).astype(np.uint16)

        # pdata_cso = pdata_cso.astype(np.uint16)

        cur_jd = ob.jd
        jdstr = str(cur_jd)
        jdsplt = jdstr.split('.')
        chip_string = 'cloud' + '_' + str(cso_counter) + '_' + str(intensity) + '_' + str(n_plumes) + '.png'

        out_file = chip_dir + chip_string
        final_file = out_file.replace('.png', '.jpg')

        image_data = pdata_cso[:, :, np.newaxis]

        if ~os.path.isfile(out_file) or ~os.path.isfile(final_file):
            numpngw.write_png(out_file, image_data)

            im = Image.open(out_file)

            im.resize((416, 416), Image.NEAREST).save(out_file)

            png_to_jpeg(out_file)

            ocount = 0
            file_lines = newline
            xlocl = list()
            ylocl = list()
            for j in range(1 + n_plumes):
                idx1 = int(j * 8)
                idx2 = int(idx1 + 8)

                if j == 0:
                    xloc = (rso_com[0] + 0.5) / 32  # + (random.random() * 0.01) - 0.005
                    yloc = (rso_com[1] + 0.5) / 32  # + (random.random() * 0.01) - 0.005
                    bbx = bbx_rso
                    bby = bby_rso
                else:
                    xloc = ((cso_com_list[j - 1][0] + 0.5) / 32)
                    yloc = ((cso_com_list[j - 1][1] + 0.5) / 32)

                    gb = np.where(pmasks[j - 1] > 0)
                    fl = np.min(gb[1])
                    fr = np.max(gb[1])
                    fd = np.min(gb[0])
                    fu = np.max(gb[0])
                    bbx = (np.abs(fl - fr)) / 32
                    bby = (np.abs(fd - fu)) / 32
                    # print('BBX: {0: 2.2f}    BBY: {1: 2.2f}'.format(bbx, bby))
                    if bbx < (1 / 32) or bby < (1 / 32):
                        bbx

                    # prev_x = xloc*32
                    # prev_y = yloc*32

                if np.all(xloc != 0) or np.all(yloc != 0):
                    xloc = np.round(xloc, 5)
                    yloc = np.round(yloc, 5)
                    bbx = np.round(bbx, 5)
                    bby = np.round(bby, 5)

                    xlocl.append(xloc)
                    ylocl.append(yloc)

                    if j == 0:
                        label_int = str(2)
                    else:
                        label_int = str(0)

                    # if j > 1:
                    #     distance = ((xloc*32 - prev_x)**2 + (yloc*32 - prev_y)**2)**0.5
                    #     if distance > 2:
                    #         temp_line = label_int + ' ' + str(yloc) + ' ' + str(xloc) + ' ' + str(bbx) + ' ' + str(bby) + '\n'
                    #         file_lines = file_lines + temp_line
                    #         ocount += 1
                    # else:
                    temp_line = label_int + ' ' + str(yloc) + ' ' + str(xloc) + ' ' + str(bbx) + ' ' + str(bby) + '\n'
                    file_lines = file_lines + temp_line
                    ocount += 1

            new_chip = final_file.replace('5017_26624', '26624_l')
            dir_parts = new_chip.split('/')
            new_dir = dir_parts[0] + '/' + dir_parts[1] + '/' + dir_parts[2] + '/' + dir_parts[3] + '/' + dir_parts[4] + '/' + dir_parts[5]
            try:
                os.mkdir(new_dir)
                # shutil.copy2('C:/RealTimeChipDetection/chips_raw/2017_10_30_l/classes.txt', new_dir + '/' + 'classes.txt')
                # shutil.copy2('C:/RealTimeChipDetection/test_chips/telkom1_l/classes.txt', new_dir + '/' + 'classes.txt')
            except:
                pass

            fname = new_chip.replace('.jpg', '.txt')
            f = open(fname, 'w')
            f.write(file_lines)
            f.close()

            shutil.copy2(final_file, new_chip)
