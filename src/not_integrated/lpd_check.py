import os
import sys

def raw_month(path):
    day_folders = [f.path for f in os.scandir(path) if f.is_dir()]
    day_folders.sort()
    cycles = []
    for day_folder in day_folders:
        aux = [os.path.basename(f.path) for f in os.scandir(day_folder) if f.is_dir()]
        aux.sort()
        cycles += aux
    return cycles


def lpd_month(path):
    cycles = [os.path.basename(f.path)[7:] for f in os.scandir(path) if f.is_dir()]
    cycles.sort()
    return cycles


def main():
    year = sys.argv[1]
    month = sys.argv[2]
    raw = raw_month(f'/raid/Salvador_raw_imgs_frames/raw/{year}/{month}-{year}')
    lpd = lpd_month(f'/raid/Salvador_raw_imgs_frames/LPD/{year}_{month}')
    miss = [cycle for cycle in raw if cycle not in lpd]
    file = open(f'./missing/missing_{year}_{month}.txt', 'w')
    print(f'{year}/{month}: {len(miss)} missing cycles')
    for cycle in miss:
        file.write(cycle + "\n")
    file.close()

if __name__ == '__main__':
    main()