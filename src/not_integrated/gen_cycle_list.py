import os
import sys
import tqdm
import datetime
import pandas as pd


def parse_cycle(cycle_str):
    tk = cycle_str.split()
    raw_date = tk[0].split('_')[1]
    date_tk = raw_date.split('-')
    for i in range(len(date_tk)):
        date_tk[i] = int(date_tk[i])
    date = datetime.date(day=date_tk[2], month=date_tk[1], year=date_tk[0])
    raw_time = tk[1][:-7]
    time_tk = raw_time.split('_')
    raw_sec = time_tk[2]
    split_sec = raw_sec.split('.')
    time_tk[2] = split_sec[0]
    time_tk.append(split_sec[1])
    for i in range(len(time_tk)):
        time_tk[i] = int(time_tk[i])
    time = datetime.time(hour=time_tk[0],
                         minute=time_tk[1],
                         second=time_tk[2],
                         microsecond=time_tk[3])
    return date, time


def main():
    path = os.path.abspath(sys.argv[1])
    lpd_path = os.path.abspath(sys.argv[2])
    years = os.listdir(path)
    years.sort()
    for year in years:
        path_y = os.path.join(path, year)
        months = os.listdir(path_y)
        months.sort()
        for month in tqdm.tqdm(months):
            path_ym = os.path.join(path_y, month)
            days = os.listdir(path_ym)
            days.sort()
            for day in days:
                path_ymd = os.path.join(path_ym, day)
                cycles = os.listdir(path_ymd)
                cycles.sort()
                for cycle in cycles:
                    cycle_list = []
                    cycle_dict = {}
                    path_ymdc = os.path.join(path_ymd, cycle)
                    lpd_cycle = f'LpdSeg_{cycle}'
                    lpd_cycle_path = os.path.join(lpd_path, year, month,
                                                  day, lpd_cycle)
                    raw_imgs = os.listdir(path_ymdc)
                    n_raw_imgs = len(raw_imgs)
                    for img in raw_imgs:
                        if img.endswith('.txt'):
                            n_raw_imgs -= 1
                            continue
                        cycle_dict[img[:23]] = 0
                    if os.path.isdir(lpd_cycle_path):
                        lpd_imgs = os.listdir(lpd_cycle_path)
                        n_lpd_imgs = len(lpd_imgs)
                        for img in lpd_imgs:
                            if img.endswith('.ppm') or img.endswith('.log') or img.endswith('.txt') or img.startswith('.'):
                                n_lpd_imgs -= 1
                                continue
                            cycle_dict[img[:23]] += 1
                    else:
                        print(f'NO LPD FOR CYCLE:{cycle}')
                        n_lpd_imgs = -1
                    cycle_list = cycle_dict.items()
                    df = pd.DataFrame(cycle_list, columns=['raw_file', 'n_rois'])
                    df.to_csv(os.path.join(path, '..', 'raw_csv_lists', f'{cycle[7:-7]}.csv'), index=False)


if __name__ == '__main__':
    main()
