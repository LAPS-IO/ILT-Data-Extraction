import os
import sys
import tqdm
import pandas as pd
from batch import create_batches, move_images
from features import compute_features, get_model
from projections import compute_projections
from data import generate_bkg, generate_thumbnails, add_scale, remove_scale, label_predictions
from aux import defaults


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
        else:
            weights_path, labels_path = '', ''
        try:
            os.mkdir(output_path, mode=0o755)
        except FileNotFoundError:
            print('Output folder is invalid, please check!')
            exit()
        except FileExistsError:
            print('Output folder already exists! Please provide a name for a new folder instead.')
            exit()
    else:
        print('Wrong number of arguments!')
        print('Usage: main.py <input_folder> <output_folder> [<model_path> <label_path > (optionals)]')
        exit()

    print('Weights:', weights_path)
    print('Labels :', labels_path, "\n")

    # Step 1: Create batches and remove scales
    df_batches = create_batches(input_path, output_path)
    move_images(input_path, df_batches, output_path)
    remove_scale(os.path.join(output_path, defaults['images']))

    model = get_model(load=True, num_classes=defaults['num_classes'])
    images_folder = os.path.join(output_path, defaults['images'])
    df_batches = pd.read_csv(os.path.join(output_path, 'batches.csv'), index_col=None)

    base_id = defaults['base_tsne_id']
    print('Computing base features...')
    features, path_images, predictions = compute_features(images_folder, base_id, model, weights_path)
    print('Computing base projections...')
    df, base_tsne = compute_projections(output_path, project_name, base_id, features, path_images, df_batches, predictions, compute_base=True, save=False)
    num_batches = len(os.listdir(images_folder))

    df_folder = os.path.join(output_path, defaults['dataframes'])
    thumbnails_folder = os.path.join(output_path, defaults['thumbnails'])

    for i in tqdm.trange(num_batches, desc='Processing batches', unit='batch', ascii=True):
        batch_id = 'batch_{:04d}'.format(i + 1)

        # Step 2: Extract data
        features, path_images, predictions = compute_features(images_folder, batch_id, model, weights_path)
        df = compute_projections(output_path, project_name, batch_id, features, path_images, df_batches, predictions, base_tsne=base_tsne)

        # Step 3: Generate CSVs + backgrounds
        generate_bkg(df_folder, images_folder, output_path, project_name, batch_id)

        # Step 4: Generate thumbnails
        input_path = os.path.join(images_folder, batch_id, defaults['inner_folder'])
        generate_thumbnails(input_path, thumbnails_folder, batch_id, defaults['thumbnails_size'])

        # Step 5: Add scale to thumbnails
        input_path = os.path.join(images_folder, batch_id, defaults['inner_folder'])
        add_scale(input_path, batch_id)

        # Step 6: Label predictions
        if not labels_path == '':
            label_predictions(df_folder, labels_path, project_name, batch_id)


if __name__ == '__main__':
    main()