import multiprocessing as mp
import os
import sys
import timeit
from datetime import timedelta

import pandas as pd
import tqdm

from aux import defaults
from batch import create_batches, symlink_images
from data import (generate_bkg, label_predictions, read_labels)
from features import compute_features, get_model
from projections import compute_projections


def update(progress_bar):
    progress_bar.update(1)


def main():
    # read argv
    if len(sys.argv) > 2:
        input_path = os.path.abspath(sys.argv[1])
        output_path = os.path.abspath(sys.argv[2])
        project_name = os.path.basename(output_path)
        if not os.path.isdir(input_path):
            print('Input folder is invalid, please check!')
            exit()
        if len(sys.argv) == 5:
            weights_path = os.path.abspath(sys.argv[3])
            if not os.path.exists(weights_path):
                print('Error! Model file not found!')
                exit()
            labels_path = os.path.abspath(sys.argv[4])
            if not os.path.exists(labels_path):
                print('Error! Label file not found!')
                exit()
            labels_dict = read_labels(labels_path)
            num_classes = len(labels_dict)
            print('Num Classes:', num_classes, "\n",'Weights:', weights_path, "\n", 'Labels:', labels_path, "\n")
        else:
            weights_path, labels_path = '', ''
            num_classes = defaults['num_classes']
        try:
            os.mkdir(output_path, mode=0o755)
        except FileNotFoundError:
            print('Output folder is invalid, please check!')
            exit()
    else:
        print('Wrong number of arguments!')
        print('Usage: main.py <input_folder> <output_folder> [<model_path> <label_path > (optionals)]')
        exit()

    # Step 1: Create batches and remove scales
    df_batches = create_batches(input_path, output_path)
    symlink_images(input_path, df_batches, output_path)
    
    model = get_model(load=True, num_classes=num_classes)
    images_folder = os.path.join(output_path, defaults['images'])
    df_batches = pd.read_csv(os.path.join(output_path, 'batches.csv'), index_col=None)

    base_id = defaults['base_tsne_id']
    print('Computing base features...')
    start = timeit.default_timer()
    features, path_images, predictions = compute_features(images_folder, base_id, model, weights_path)
    end = timeit.default_timer()
    print('Total time:', timedelta(seconds=(end - start)), "\n")

    print('Computing base projections...')
    start = timeit.default_timer()
    base_tsne = compute_projections(output_path, project_name, base_id, features, path_images, df_batches, predictions, compute_base=True, save=False)
    end = timeit.default_timer()
    print('Total time:', timedelta(seconds=(end - start)), "\n")

    num_batches = len(os.listdir(images_folder))
    df_folder = os.path.join(output_path, defaults['dataframes'])

    # Step 2: Extract data
    print('Computing all features/projections...')
    for i in tqdm.tqdm(range(0, num_batches), ascii=True, ncols=79, unit='bat'):
        batch_id = 'batch_{:04d}'.format(i + 1)
        features, path_images, predictions = compute_features(images_folder, batch_id, model, weights_path)
        compute_projections(output_path, project_name, batch_id, features, path_images, df_batches, predictions, base_tsne=base_tsne)
    print()

    # Step 3: Generate CSVs + backgrounds
    print('Generating backgrounds...')
    start = timeit.default_timer()
    backgrounds_folder = os.path.join(output_path, defaults['backgrounds'])
    if not os.path.isdir(backgrounds_folder):
        os.mkdir(backgrounds_folder, mode=0o755)

    with tqdm.tqdm(range(0, num_batches), ascii=True, ncols=79, unit='bat') as pbar:
        with mp.Pool(mp.cpu_count()) as pool:
            for i in range(num_batches):
                pool.apply_async(generate_bkg, callback=update(pbar), args=(backgrounds_folder, df_folder, images_folder, project_name, i + 1))
            pool.close()
            pool.join()
    end = timeit.default_timer()
    print('Total time:', timedelta(seconds=(end - start)), "\n")

    # Step 4: Label predictions
    if labels_path != '':
        print('Labeling predictions...')
        full_df = pd.DataFrame()
        start = timeit.default_timer()
        for i in tqdm.tqdm(range(0, num_batches), ascii=True, ncols=79, unit='bat'):
            batch_id = 'batch_{:04d}'.format(i + 1)
            df_path = os.path.join(df_folder, batch_id + '_' + project_name + '.csv')
            batch_df = pd.read_csv(df_path)
            label_predictions(batch_df, labels_path)
            full_df = pd.concat([full_df, batch_df[["names", "pred"]]], ignore_index=True)
            batch_df.to_csv(df_path, index=False)
        full_df.to_csv(os.path.join(output_path, 'complete.csv'), index=False)
        end = timeit.default_timer()
        print('Total time:', timedelta(seconds=(end - start)))


if __name__ == '__main__':
    main()
