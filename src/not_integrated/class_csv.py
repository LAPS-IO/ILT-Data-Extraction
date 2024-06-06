import os
import sys
import tqdm
import shutil
import pandas as pd


def main():
    fito = pd.read_csv(os.path.abspath(sys.argv[1]))
    dest = '/raid/fito_copia_2021_06/images'
    for t in tqdm.tqdm(fito.itertuples(), unit='img'):
        img = f'/raid/Salvador_raw_imgs_frames/LPD/2021_06/{t[3]}/{t[2]}'
        shutil.copy2(img, dest)

if __name__ == '__main__':
    main()