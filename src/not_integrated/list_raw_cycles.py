import os
import tqdm
import pandas as pd


def list_dirs(base):
    path = os.path.join(base)
    dirs = os.listdir(path)
    dirs = [d for d in dirs if os.path.isdir(os.path.join(path, d))]
    return dirs


def list_raws():
    print('Listing raws...')
    basepath = '/raid/Salvador_raw_imgs_frames/raw'
    raw_set = set()
    years = list_dirs(basepath)
    for year in years:
        year_path = os.path.join(basepath, year)
        months = list_dirs(year_path)
        for month in tqdm.tqdm(months):
            month_path = os.path.join(year_path, month)
            days = list_dirs(month_path)
            for day in days:
                day_path = os.path.join(month_path, day)
                cycles = list_dirs(day_path)
                for cycle in cycles:
                    raw_set.add(cycle[7:-7] + '\n')
    return raw_set


def list_lpds():
    print('Listing lpds...')
    basepath = '/raid/Salvador_raw_imgs_frames/LPD'
    lpd_set = set()
    years = list_dirs(basepath)
    for year in years:
        year_path = os.path.join(basepath, year)
        months = list_dirs(year_path)
        for month in tqdm.tqdm(months):
            month_path = os.path.join(year_path, month)
            days = list_dirs(month_path)
            for day in days:
                day_path = os.path.join(month_path, day)
                cycles = list_dirs(day_path)
                for cycle in cycles:
                    lpd_set.add(cycle[14:-7] + '\n')
    return lpd_set


def main():
    raw_set = list_raws()
    lpd_set = list_lpds()
    raw_lpd = raw_set - lpd_set
    lpd_raw = lpd_set - raw_set
    raw_lpd = list(raw_lpd)
    raw_lpd.sort()
    lpd_raw = list(lpd_raw)
    lpd_raw.sort()
    print('raw_lpd')
    with open('raw_lpd.txt', 'w') as f:
        f.writelines(raw_lpd)
    print('lpd_raw')
    with open('lpd_raw.txt', 'w') as f:
        f.writelines(lpd_raw)


if __name__ == '__main__':
    main()
