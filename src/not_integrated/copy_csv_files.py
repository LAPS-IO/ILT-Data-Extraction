import os
import sys
import shutil

import pandas as pd
import tqdm


def image_list(input_path):
    print('Generating image list for:', input_path)
    imgs = {}
    for pwd, children, files in tqdm.tqdm(os.walk(input_path)):
        for file in files:
            imgs[file] = os.path.join(pwd, file)
    return imgs


def main():
    all_imgs = image_list(os.path.abspath(sys.argv[2]))
    print(all_imgs[0])
    print('Filtering images in:', sys.argv[1])
    for pwd, chd, csvs in os.walk(sys.argv[1]):
        for csv in csvs:
            print(csv)
            filtered_imgs = []
            df = pd.read_csv(pwd + '/' + csv)
            for t in df.itertuples():
                print(t)
                filtered_imgs.append(all_imgs[t[2]])

            print('Writing filtered images to:', os.path.abspath(sys.argv[3]))
            # file = open(sys.argv[3], 'w')
            count = 0
            for fi in tqdm.tqdm(filtered_imgs, unit='img'):
                count += 1
                shutil.copy2(fi, os.path.abspath(sys.argv[3] + '/' + csv[:-4]))
                if count == 10000:
                    break


if __name__ == '__main__':
    main()
