import math
import os
import timeit
import shutil
import pandas as pd
import multiprocessing as mp
from aux import defaults
from datetime import timedelta


# Input:
# (1) input_path: a string containing the path to the input dataset
# (2) ouput_path: a string containing the path to the output folder
# Output:
# (1) a Pandas DataFrame mapping each image to its original class
#     and its assigned batch number
def create_batches(input_path, output_path):
    print('Creating batches...')
    imgs = []
    if os.path.isdir(input_path):
        for pwd, children, files in os.walk(input_path):
            imgs += [ (file, os.path.basename(pwd)) for file in files if (file.endswith('.png') or file.endswith('.jpg')) ]
    else:
        img_list = open(input_path, 'r')
        for img in img_list:
            img = img.strip()
            imgs.append((os.path.basename(img), os.path.dirname(img)))
        img_list.close()

    num_batches = math.ceil(len(imgs) / defaults['BATCH_MAX_SIZE'])
    images_per_batch = math.ceil(len(imgs) / num_batches)

    df = pd.DataFrame(imgs, columns=['names', 'klass'])
    df = df.sample(frac=1).reset_index(drop=True)

    batches = []
    count = df.shape[0]

    i = 1
    while count > images_per_batch:
        batches += [f'batch_{i:04d}'] * images_per_batch
        count -= images_per_batch
        i += 1
    batches += [f'batch_{i:04d}'] * count

    df['batch'] = batches
    df.to_csv(os.path.join(output_path, 'batches.csv'), index=None)
    print("Done creating batches!\n")
    return df


def symlink_batch_images(input_path, images_folder, df):
    for row in df.itertuples():
        batch_outer_folder = os.path.join(images_folder, row.batch)
        if not os.path.isdir(batch_outer_folder):
            os.mkdir(batch_outer_folder, mode=0o755)

        batch_folder = os.path.join(batch_outer_folder, defaults['inner_folder'])
        if not os.path.isdir(batch_folder):
            os.mkdir(batch_folder, mode=0o755)

        original_path = os.path.join(input_path, row.klass, row.names)
        try:
            match defaults['img_mode']:
                case 'ln':
                    os.symlink(original_path, os.path.join(batch_folder, row.names))
                case 'cp':
                    shutil.copy2(original_path, os.path.join(batch_folder, row.names))
                case 'mv':
                    shutil.move(original_path, os.path.join(batch_folder, row.names))
                case _:
                    print('Wrong img_mode flag. Check src/aux.py file.')
                    exit()
        except:
            df.drop(row.Index, inplace=True)


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
def symlink_images(input_path, df, dataset_path):
    images_folder = os.path.join(dataset_path, defaults['images'])
    os.mkdir(images_folder, mode=0o755)

    print(f'Creating image symlinks to {dataset_path}')
    start = timeit.default_timer()
    groups = [splitted_df for _, splitted_df in df.groupby(df.batch)]
    args = [(input_path, images_folder, group) for group in groups]
    with mp.Pool(mp.cpu_count()) as pool:
        result = pool.starmap_async(symlink_batch_images, args)
        result.wait()
    end = timeit.default_timer()
    print(f'Total time: {timedelta(seconds=(end - start))}\n')
