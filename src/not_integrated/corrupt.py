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
    imgs = []
    for pwd, children, files in os.walk(input_path):
        rel_pwd = pwd[len(input_path) + 1:]
        imgs += [ os.path.join(pwd, file) for file in files if (file.endswith('.png') or file.endswith('.jpg')) ]
    print('Found', len(imgs), "images.\n")

    print('Looking for corrupt images in...')
    corrupt_list = open("%s.txt" % (os.path.basename(input_path)), 'w')
    corrupt_list_base = []
    corrupt_imgs = 0
    for img_path in tqdm.tqdm(imgs, unit='img'):
        try:
            img = Image.open(img_path)
        except:
            print('Error processing:', img_path)
            img.close()
            corrupt_imgs += 1
            os.remove(img_path)
            corrupt_list.write(img_path)
            corrupt_list_base.append(os.path.basename(img_path))
        else:
            img.close()
    print('Found', corrupt_imgs, 'corrupt images!')
    return corrupt_list_base


if __name__ == '__main__':
    corr_list = main(os.path.abspath(sys.argv[1]))
    if len(argv) == 3:
        clean_csv(os.path.abspath(sys.argv[2]), corr_list)