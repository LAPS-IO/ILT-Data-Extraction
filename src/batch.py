import os
import pandas as pd
import math
import shutil
import PIL
import tqdm
from aux import defaults


# Input:
# (1) input_path: a string containing the path to the input dataset
# (2) ouput_path: a string containing the path to the output folder
# Output:
# (1) a Pandas DataFrame mapping each image to its original class
#     and its assigned batch number
def create_batches(input_path, output_path):
    print('Creating batches...')
    imgs = []
    for pwd, children, files in os.walk(input_path):
        rel_pwd = pwd[len(input_path) + 1:]
        imgs += [ (file, rel_pwd) for file in files if (file.endswith('.png') or file.endswith('.jpg')) ]

    num_batches = math.ceil(len(imgs) / defaults['BATCH_MAX_SIZE'])
    images_per_batch = math.ceil(len(imgs) / num_batches)

    df = pd.DataFrame(imgs, columns=['names', 'klass'])
    df = df.sample(frac=1).reset_index(drop=True)

    batches = []
    count = df.shape[0]

    i = 1
    while count > images_per_batch:
        batches.extend(['batch_{:04d}'.format(i)] * images_per_batch)
        count -= images_per_batch
        i += 1
    batches.extend(['batch_{:04d}'.format(i)] * count)

    df['batch'] = batches
    df.to_csv(os.path.join(output_path, 'batches.csv'), index=None)
    print('Done creating batches!')
    return df


# Inputs:
  # (1) input_path: a string containing the path to the input dataset
  # (2) df: a Pandas DataFrame mapping images into batch numbers
  # (3) dataset_path: a string containing the path to the output folder
# Side effects:
  # (1) creates a folder called images inside the previous directory
  # (2) creates multiples batch_XXXX folders inside the images directory,
  #     where XXXX is the id of the folder
  # (3) moves the images from the input_path into their respective batches
# Output:
  # None
def move_images(input_path, df, dataset_path):
    images_folder = os.path.join(dataset_path, defaults['images'])
    os.mkdir(images_folder, mode=0o755)

    print('Moving images to ' + dataset_path)
    with tqdm.trange(df.shape[0], desc='Images moved', unit=" images", ascii=True) as pbar:
        for row in df.itertuples():
            batch_outer_folder = os.path.join(images_folder, row.batch)
            if not os.path.isdir(batch_outer_folder):
                os.mkdir(batch_outer_folder, mode=0o755)

            batch_folder = os.path.join(batch_outer_folder, defaults['inner_folder'])
            if not os.path.isdir(batch_folder):
                os.mkdir(batch_folder, mode=0o755)

            original_path = os.path.join(input_path, row.klass, row.names)

            if os.stat(original_path).st_size == 0:
                print('Warning: ', original_path, 'is not a valid image! Skipping...')
                df.drop(row.Index, inplace=True)
            else:
                shutil.move(original_path, os.path.join(batch_folder, row.names))
            pbar.update(1)