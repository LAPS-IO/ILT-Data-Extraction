import os
import sys
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


def main():
    roi_filename = os.path.basename(sys.argv[1])[:-4]
    cycle_list_filename = '/raid/Salvador_raw_imgs_frames/raw/cycles.csv'
    roi_data = parse_roi_filename(roi_filename)
    cycle_df = pd.read_csv(cycle_list_filename)
    dates = cycle_df['date'].tolist()
    times = cycle_df['time'].tolist()
    dt_list = []
    for date, time in zip(dates, times):
        dts = f'{date} {time}'
        dt = datetime.datetime.strptime(dts, '%Y-%m-%d %H:%M:%S.%f')
        dt_list.append(dt)
    roi_cycle_index = bisect.bisect(dt_list, roi_data['datetime']) - 1
    print(f'ROI  : {roi_data['datetime']} {roi_data['xy']}')
    print(f'CICLO: {dt_list[roi_cycle_index]} idx: {roi_cycle_index}')


if __name__ == '__main__':
    main()
