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
    files, klasses = [], []
    for klass in os.listdir(input_path):
        klass_path = os.path.join(input_path, klass)
        if os.path.isdir(klass_path):
            for file in os.listdir(klass_path):
                files.append(file)
                klasses.append(klass)

    num_batches = math.ceil(len(files) / defaults['BATCH_MAX_SIZE'])
    images_per_batch = math.ceil(len(files) / num_batches)

    df = pd.DataFrame(list(zip(files, klasses)), columns=['names', 'klass'])
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


def move_images(input_path, df, dataset_path, check_valid=False):
    images_folder = os.path.join(dataset_path, defaults['images'])
    os.mkdir(images_folder, mode=0o755)

    print('Moving images to ' + dataset_path)
    for row in tqdm.tqdm(df.itertuples(), desc='Images moved', unit=" images", ascii=True):
        batch_outer_folder = os.path.join(images_folder, row.batch)
        if not os.path.isdir(batch_outer_folder):
            os.mkdir(batch_outer_folder, mode=0o755)

        batch_folder = os.path.join(batch_outer_folder, defaults['inner_folder'])
        if not os.path.isdir(batch_folder):
            os.mkdir(batch_folder, mode=0o755)

        original_path = os.path.join(input_path, row.klass, row.names)

        try:
            if check_valid:
                PIL.Image.open(original_path)
            shutil.copy2(original_path, os.path.join(batch_folder, row.names))
        except PIL.UnidentifiedImageError:
            print('Warning: ', original_path, 'is not a valid image')

    print("Finished moving the images.\n")