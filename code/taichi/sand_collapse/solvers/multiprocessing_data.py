from multiprocessing.dummy import freeze_support
import time
import multiprocessing
import dry_sand_collapse_quadratic_MLS_MPM_plane_strain

# def sleepy_man():
#     print('Starting to sleep')
#     time.sleep(1)
#     print('Done sleeping')


if __name__ == '__main__':
    freeze_support()

    for i in range(10000):

        tic = time.time()
        dry_sand_collapse_quadratic_MLS_MPM_plane_strain
        toc = time.time()

        print('Done in {:.4f} seconds'.format(toc-tic))