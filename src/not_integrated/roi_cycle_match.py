import os
import sys
import tqdm
import bisect
import datetime
import pandas as pd


def parse_date(raw_date):
    split_date = raw_date.split('-')
    dt = [int(sd) for sd in split_date]
    date = datetime.date(year=dt[0], month=dt[1], day=dt[2])
    return date


def parse_time(raw_time):
    split_sec = raw_time[2].split('.')
    time_tokens = raw_time[0:2] + split_sec
    ft = [int(tt) for tt in time_tokens]
    time = datetime.time(hour=ft[0],
                         minute=ft[1],
                         second=ft[2],
                         microsecond=ft[3] * 1000)
    return time


def parse_xy(raw_xy):
    xy = (int(raw_xy[0]), int(raw_xy[1]))
    return xy


def parse_roi_filename(roi_filename):
    raw_tokens = roi_filename.split('_')
    raw_date = raw_tokens[0]
    raw_time = raw_tokens[1:4]
    raw_xy = raw_tokens[4:]
    date = parse_date(raw_date)
    time = parse_time(raw_time)
    dt = datetime.datetime.combine(date, time)
    roi_data = {
        'datetime': dt,
        'xy': parse_xy(raw_xy)
    }
    return roi_data


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
    cycle_list_filename = '/raid/Salvador_raw_imgs_frames/raw/cycles.csv'
    cycle_df = pd.read_csv(cycle_list_filename)
    dates = cycle_df['date'].tolist()
    times = cycle_df['time'].tolist()
    dt_list = []
    for date, time in zip(dates, times):
        dts = f'{date} {time}'
        dt = datetime.datetime.strptime(dts, '%Y-%m-%d %H:%M:%S.%f')
        dt_list.append(dt)
    full_list = []
    days = os.listdir(path)
    days.sort()
    for day in tqdm.tqdm(days):
        path_d = os.path.join(path, day)
        cycles = os.listdir(path_d)
        cycles.sort()
        for cycle in cycles:
            lpd_cycle_path = os.path.join(path_d, cycle)
            if os.path.isdir(lpd_cycle_path):
                lpd_imgs = os.listdir(lpd_cycle_path)
                for img in lpd_imgs:
                    if img.endswith('ppm') or img.endswith('log') or img.endswith('txt'):
                        continue
                    elif img.startswith('.'):
                        print(img)
                        continue
                    else:
                        roi_data = parse_roi_filename(img[:-4])
                        roi_cycle_index = bisect.bisect(dt_list, roi_data['datetime']) - 1
                        roi_tuple = (img, dt_list[roi_cycle_index])
                        full_list.append(roi_tuple)
            else:
                continue
    df = pd.DataFrame(full_list, columns=['roi', 'cycle'])
    df.to_csv(os.path.join(path, 'roi_cycle.csv'), index=False)


if __name__ == '__main__':
    main()
