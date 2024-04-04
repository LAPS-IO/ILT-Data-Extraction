import os
import sys

import pandas as pd
import tqdm
from PIL import Image


def clean_csv(csv_path, img_list):
    df = pd.read_csv(csv_path)
    df.drop(img_list, inplace=True)
    df.to_csv(csv_path, index=None)


def main(input_path):
    print('Generating image list for:', input_path)
    txts, imgs = [], []
    for pwd, children, files in os.walk(input_path):
        rel_pwd = pwd[len(input_path) + 1:]
        for file in files:
            if file.endswith('.txt'):
                txts.append(os.path.join(pwd, file))
            elif file.endswith('.png') or file.endswith('.jpg'):
                imgs.append(os.path.join(pwd, file))
    print('Found', len(txts), 'text files.')
    print('Found', len(imgs), 'images.')

    print('Writing corrupted txts...')
    file_list = open("%s.txt" % (os.path.basename(input_path)), 'w')
    file_list.write("txt\n")
    file_list.write(str(len(txts)) + "\n")
    for txt_path in txts:
        os.remove(txt_path)
        file_list.write(txt_path + "\n")

    print('Looking for corrupted images...')
    file_list.write("img\n")
    corrupted = []
    for img_path in tqdm.tqdm(imgs, unit='img'):
        if os.stat(img_path).st_size == 0:
            os.remove(img_path)
            corrupted.append(img_path)

    print('Writing corrupted imgs...')
    print('Found', len(corrupted), 'images.')
    file_list.write(str(len(corrupted)) + "\n")
    for cor_path in corrupted:
        file_list.write(cor_path + "\n")
    file_list.close()
    return corrupted


if __name__ == '__main__':
    corr_list = main(os.path.abspath(sys.argv[1]))
    if len(sys.argv) == 3:
        clean_csv(os.path.abspath(sys.argv[2]), corr_list)