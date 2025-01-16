import tqdm
import sys
import os
from pathlib import Path


def main():
    filename = sys.argv[1]
    tokens = filename.split('_')
    basepath = os.path.join('/raid', 'Salvador_raw_imgs_frames', 'raw')
    dest_path = os.path.join('/raid', 'Salvador_raw_imgs_frames', 'TODO', f'{tokens[6]}_{tokens[7]}')
    try:
        os.mkdir(dest_path)
    except:
        pass
    monthpath = os.path.join(basepath, tokens[6], tokens[7])
    file = open(os.path.abspath(filename), 'r')
    for line in tqdm.tqdm(file):
        line_tokens1 = line.split(' ')
        line_tokens2 = line_tokens1[0].split('-')
        day = line_tokens2[2]
        src_cycle_path = os.path.join(monthpath, day, f'Basler_{line[:-1]}_frames')
        dst_cycle_path = os.path.join(dest_path, os.path.basename(src_cycle_path))
        Path(dst_cycle_path).symlink_to(src_cycle_path)


if __name__ == '__main__':
    main()
