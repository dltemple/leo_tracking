import os
import sys
import glob
import time
# from astride import Streak

if __name__ == '__main__':

    # base_path = os.path.dirname(os.path.realpath(__file__)).replace('src', '')
    # image_folder = os.path.join(base_path, 'data')
    #
    # file_list = glob.glob(os.path.join(image_folder, '*/*/*/*.jpeg'))
    # # file_list = [os.path.join(image_folder, x)
    # #              for x in file_list if x.endswith(('.fits'))]
    # print(len(file_list))
    # for file in file_list:
    #     os.rename(file, file.replace('.jpeg', '.jpg'))
    #
    # exit()

    base_path = os.path.dirname(os.path.realpath(__file__)).replace('src', '')
    image_folder = os.path.join(base_path, 'data/2020-02-23/sat2')

    file_list = os.listdir(image_folder)
    file_list = [os.path.join(image_folder, x)
                 for x in file_list if x.endswith(('.fits'))]

    for file in file_list:
        t0 = time.time()
        print('Beginning File :: {}'.format(file))
        streak = Streak(file, remove_bkg='map')

        streak.detect()

        print('Finished. Time :: {0:4.4f}'.format(time.time()-t0))

        streak.write_outputs()
        streak.plot_figures()

    # create_gif(file_list, duration=0.2, output_file=output_file)

    exit()
