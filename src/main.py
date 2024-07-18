import os
import sys
import datetime
import timeit
import pandas as pd
import tqdm
from aux import defaults
from batch import create_batches, symlink_images
from data import ( label_predictions, read_labels)
from features import compute_features, get_model
from projections import compute_projections


def read_paths(argv):
    input_path = os.path.abspath(argv[1])
    output_path = os.path.abspath(argv[2])
    try:
        os.mkdir(output_path, mode=0o755)
    except FileNotFoundError:
        print('Output folder is invalid, please check!')
        exit()
    except FileExistsError:
        print('Output folder already exists, please check!')
        exit()
    return input_path, output_path


def read_network(argv):
    weights_path = os.path.abspath(sys.argv[3])
    labels_path = os.path.abspath(sys.argv[4])
    if not os.path.exists(weights_path):
        print('Error! Model file not found!')
        exit()
    if not os.path.exists(labels_path):
        print('Error! Label file not found!')
        exit()
    return weights_path, labels_path


def log(file, string):
    file.write(string)
    file.flush()


def main():
    input_path, output_path = read_paths(sys.argv)
    project_name = os.path.basename(output_path)
    log_file = open(os.path.join(output_path, f'{project_name}.log'), 'w')
    log(log_file, f'{datetime.datetime.now()}: START {project_name}\n')
    match len(sys.argv):
        case 3:
            weights_path, labels_path = '', ''
            num_classes = defaults['num_classes']
        case 5:
            weights_path, labels_path = read_network(sys.argv)
            labels_dict = read_labels(labels_path)
            num_classes = len(labels_dict)
            print(f'Num Classes: {num_classes}')
            print(f'Weights: {weights_path}')
            print(f'Labels: {labels_path}')
        case _:
            print('Wrong number of arguments!')
            print('Usage: main.py <input_folder> <output_folder> [<model_path> <label_path > (optionals)]')
            exit()

    # Step 1: Create batches and remove scales
    df_batches = create_batches(input_path, output_path)
    log(log_file, f'{datetime.datetime.now()}: Symlinks\n')
    if input_path[-4:] == '.txt':
        input_path = f'/raid/Salvador_raw_imgs_frames/LPD/{project_name}'
    symlink_images(input_path, df_batches, output_path)
    
    model = get_model(load=True, num_classes=num_classes)
    images_folder = os.path.join(output_path, defaults['images'])
    df_batches = pd.read_csv(os.path.join(output_path, 'batches.csv'), index_col=None)

    base_id = defaults['base_tsne_id']
    log(log_file, f'{datetime.datetime.now()}: Base features\n')
    print('Computing base features...')
    start = timeit.default_timer()
    features, path_images, predictions = compute_features(images_folder, base_id, model, weights_path)
    end = timeit.default_timer()
    print('Total time:', datetime.timedelta(seconds=(end - start)), "\n")

    log(log_file, f'{datetime.datetime.now()}: Base projections\n')
    print('Computing base projections...')
    start = timeit.default_timer()
    base_tsne = compute_projections(
        output_path, project_name, base_id, features, path_images,
        df_batches, predictions, compute_base=True, save=False)
    end = timeit.default_timer()
    print('Total time:', datetime.timedelta(seconds=(end - start)), "\n")

    num_batches = len(os.listdir(images_folder))
    df_folder = os.path.join(output_path, defaults['dataframes'])

    # Step 2: Extract data
    log(log_file, f'{datetime.datetime.now()}: All feat/proj\n')
    print('Computing all features/projections...')
    for i in tqdm.tqdm(range(0, num_batches), unit='batch'):
        batch_id = f'batch_{i + 1:04d}'
        features, path_images, predictions = compute_features(images_folder, batch_id, model, weights_path)
        compute_projections(
            output_path, project_name, batch_id, features, path_images,
            df_batches, predictions, base_tsne=base_tsne)
    print()

    # Step 4: Label predictions
    if labels_path != '':
        log(log_file, f'{datetime.datetime.now()}: Labeling\n')
        print('Labeling predictions...')
        full_df = pd.DataFrame()
        start = timeit.default_timer()
        for i in tqdm.tqdm(range(0, num_batches), unit='batch'):
            batch_id = f'batch_{i + 1:04d}'
            df_path = os.path.join(df_folder, batch_id + '_' + project_name + '.csv')
            batch_df = pd.read_csv(df_path)
            label_predictions(batch_df, labels_path)
            full_df = pd.concat([full_df, batch_df[["names", "pred"]]], ignore_index=True)
            batch_df.to_csv(df_path, index=False)
        full_df.to_csv(os.path.join(output_path, 'complete.csv'), index=False)
        end = timeit.default_timer()
        print('Total time:', datetime.timedelta(seconds=(end - start)))
    log(log_file, f'{datetime.datetime.now()}: END {project_name}\n')
    log_file.close()


if __name__ == '__main__':
    main()
