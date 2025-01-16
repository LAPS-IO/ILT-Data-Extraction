import re
import os
import sys
import tqdm
import shutil


def main():
    path = os.path.abspath(sys.argv[1])
    cycles = os.listdir(path)
    for cycle in tqdm.tqdm(cycles):
        if cycle[:3] != 'Lpd':
            continue
        cycle_path = os.path.join(path, cycle)
        tokens = re.split('_|-| ', cycle)
        day = tokens[4]
        day_folder = os.path.join(path, day)
        if not os.path.exists(day_folder):
            os.mkdir(day_folder, mode=0o755)
        shutil.move(cycle_path, day_folder)


if __name__ == '__main__':
    main()
