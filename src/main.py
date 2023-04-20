import sys
from batch import create_batches, move_images
from aux import defaults
from features import compute_features, get_model
import os
from os import listdir
from os.path import basename, isdir, join, exists
from projections import compute_projections
from data import generate_data, generate_thumbnails, add_scale
import pandas as pd

def print_choices():
    print('1: Create batches')
    print('2: Extract data from batches')
    print('3: Generate CSVs + backgrounds')
    print('4: Generate thumbnails')
    print('5: Add scale to images')

def print_projects():
    output_folder = defaults['output_folder']
    projects = [f for f in listdir(output_folder) if isdir(join(output_folder, f))]
    print('List of projects:')
    print(projects)
    return projects

def choose_option():
    print_choices()
    val = input('Choose an option: \n')
    while(val not in ['1', '2', '3', '4', '5']):
        print(val, 'is not a valid choice.')
        val = input('Choose an option: \n')
    return int(val)

def choose_project():
    list_projects = print_projects()
    project_name = input('Type the name of the project: ')
    while project_name not in list_projects:
        print('Error! Project', project_name, 'does not exist.')
        project_name = input('Type the name of the project: ')
    return project_name

def choose_batch_start(project_name, images_folder):
    num_batches = len(listdir(images_folder))
    print(num_batches, 'batches', 'found in', project_name)

    batch_start = int(input('Type the number of the first batch to be processed: ') )
    while batch_start <= 0:
        print('Error! Batch', batch_start,'does not exist.')
        batch_start = int(input('Type the number of the first batch to be processed: ') )

    return batch_start

def choose_batch_end(images_folder, batch_start):
    num_batches = len(listdir(images_folder))
    batch_end = int(input('Type the number of the last batch to be processed: ') )
    while batch_end <= 0 or batch_end < batch_start:
        if batch_end <= 0:
            print('Error! Batch', batch_end,'does not exist.')
        else:
            print('Error! Starting batch is higher than the last batch.')
        batch_end = int(input('Type the number of the last batch to be processed: ') )
    return batch_end


def new_main():
    # read argv
    if len(sys.argv) != 4:
        print('Missing arguments!')
        print('Usage: main.py <project name> <input_folder> <output_folder>')
        exit()

    project_name = sys.argv[1]
    input_path = os.path.abspath(sys.argv[2])
    output_path = os.path.abspath(sys.argv[3])
    if os.path.isdir(input_path) is False:
        print('Input folder is invalid, please check!')
        exit()
    try:
        os.mkdir(output_path, mode=0o755)
    except FileNotFoundError:
        print('Output folder is invalid, please check!')
        exit()
    except FileExistsError:
        pass

    # create batches.csv
    df_batches = create_batches(input_path, output_path)

    # move images
    move_images(input_path, df_batches, output_path)

    # TODO:
    # Step 2: Extract data from batches
    # Step 3: Generate CSVs + backgrounds
    # Step 4: Generate thumbnails
    # Step 5: Add scale to images

def main():
    val = choose_option()
    if val == 1:
        input_path = input('Type the complete path to the folder with the images: ')
        while not isdir(input_path):
            print('Error!', input_path, 'is not a directory.')
            input_path = input('Type the complete path to the folder with the images: ')
        else:
            df_batches = create_batches(input_path)
            project_name = basename(input_path)
            move_images(input_path, df_batches, project_name)
            df_batches.to_csv(join(defaults['output_folder'], project_name, 'batches.csv'), index=None)

    elif val == 2:
        project_name = choose_project()
        weights_path = input('Type the complete path to the trained model (or press Enter to load the default weights): ')

        if len(weights_path) > 0 and not exists(weights_path):
            print('Error! Model not found')
        else:
            model = get_model(load = True, num_classes = defaults['num_classes'])


        images_folder = join(defaults['output_folder'], project_name, defaults['images'])

        batch_start = choose_batch_start(project_name, images_folder)
        batch_end = choose_batch_end(images_folder, batch_start)

        df_batches = pd.read_csv(join(defaults['output_folder'], project_name, 'batches.csv'), index_col = None)

        base_id = defaults['base_tsne_id']
        features, path_images = compute_features(images_folder, base_id, model, weights_path)
        df, base_tsne = compute_projections(project_name, base_id, features, path_images, df_batches, compute_base = True, save=False)

        for i in range(batch_start, batch_end + 1):
            batch_id = 'batch_{:04d}'.format(i)
            print('Processing', batch_id)
            features, path_images = compute_features(images_folder, batch_id, model, weights_path)
            df = compute_projections(project_name, batch_id, features, path_images, df_batches, base_tsne = base_tsne)

    elif val == 3:
        project_name = choose_project()
        images_folder = join(defaults['output_folder'], project_name, defaults['images'])

        batch_start = choose_batch_start(project_name, images_folder)
        batch_end = choose_batch_end(images_folder, batch_start)
        dataframes_folder = join(defaults['output_folder'], project_name, defaults['dataframes'])

        for i in range(batch_start, batch_end + 1):
            batch_id = 'batch_{:04d}'.format(i)
            print('Processing', batch_id)

            df = pd.read_csv(join(dataframes_folder, batch_id + '_' + project_name + '.csv'), index_col=None)
            generate_data(df, images_folder, project_name, batch_id)

    elif val == 4:
        project_name = choose_project()
        images_folder = join(defaults['output_folder'], project_name, defaults['images'])
        thumbnails_folder = join(defaults['output_folder'], project_name, defaults['thumbnails'])

        batch_start = choose_batch_start(project_name, images_folder)
        batch_end = choose_batch_end(images_folder, batch_start)

        for i in range(batch_start, batch_end + 1):
            batch_id = 'batch_{:04d}'.format(i)
            print('Processing', batch_id)
            input_path = join(images_folder, batch_id, defaults['inner_folder'])
            generate_thumbnails(input_path, thumbnails_folder, batch_id, defaults['thumbnails_size'])

    elif val == 5:
        project_name = choose_project()
        images_folder = join(defaults['output_folder'], project_name, defaults['images'])

        batch_start = choose_batch_start(project_name, images_folder)
        batch_end = choose_batch_end(images_folder, batch_start)

        for i in range(batch_start, batch_end + 1):
            batch_id = 'batch_{:04d}'.format(i)
            print('Processing', batch_id)
            input_path = join(images_folder, batch_id, defaults['inner_folder'])
            add_scale(input_path, batch_id)

if __name__ == '__main__':
    main()
