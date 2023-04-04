from os import listdir
from os.path import join, isdir
import pandas as pd
import math
from shutil import copy2
from aux import defaults, create_dir
import PIL

# Input:
# (1) input_path: a string containing the path to the input dataset
# Output:
# (1) a Pandas DataFrame mapping each image to its original class
#     and its assigned batch number


def create_batches(input_path):
    files = []
    classes = []
    for c in listdir(input_path):
        class_path = join(input_path, c)
        if isdir(class_path):
            for f in listdir(class_path):
                files.append(f)
                classes.append(c)

    num_batches = math.ceil(len(files)/defaults['BATCH_MAX_SIZE'])
    images_per_batch = math.ceil(len(files)/num_batches)

    df = pd.DataFrame(list(zip(files, classes)), columns=['names', 'class'])

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
    return df

# Inputs:
  # (1) input_path: a string containing the path to the input dataset
  # (2) df: a Pandas DataFrame mapping images into batch numbers
# Side effects:
  # (1) creates a folder in the output directory with the dataset name
  # (2) creates a folder called images inside the previous directory
  # (3) creates multiples batch_XXXX folders inside the images directory,
  #     where XXXX is the id of the folder
  # (4) copies the images from the input_path into their respective batches
# Output:
  # (1) True if the dataset folder with the batches was successfully created
  #     inside the output directory. Returns False otherwise.


def move_images(input_path, df, dataset_name, debug=True):
    dataset_path = join(defaults['output_folder'], dataset_name)
    if create_dir(dataset_path, ignore=False):
        images_folder = join(dataset_path, defaults['images'])
        create_dir(images_folder)
        print('Copying images to ' + dataset_path)
        div = df.shape[0]//10
        for index, row in df.iterrows():
            image_name = row['Image']
            batch_id = row['Batch']
            c = row['Class']
            batch_folder = join(images_folder, batch_id)
            create_dir(batch_folder)
            original_path = join(input_path, c, image_name)

            try:
                img = PIL.Image.open(original_path)
                copy2(original_path, join(batch_folder, image_name))

            except PIL.UnidentifiedImageError:
                print('Warning: ', original_path, 'is not a valid image')

            if (index + 1) % div == 0 and debug:
              print(str(index + 1) + '/' + str(df.shape[0]) + ' complete')

        print('Finished copying the images.')
        return images_folder
    else:
        print('Error! Output folder ' + join(defaults['output_folder'], dataset_name) + ' already exists!')
        return ''
