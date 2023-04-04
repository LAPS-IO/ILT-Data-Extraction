import sys
from batch import create_batches, move_images
from aux import defaults
from features import compute_features 
from os import listdir
from os.path import basename, isdir, join
from features import get_model
from projections import compute_projections
from data import generate_data
import pandas as pd

def print_choices():
    print('1: Create batches')
    print('2: Extract data from batches')

def print_projects():
    output_folder = defaults['output_folder']
    projects = [f for f in listdir(output_folder) if isdir(join(output_folder, f))]
    print('List of projects:')
    print(projects)
    return projects

def choose_option():
    print_choices()
    val = input('Choose an option: \n') 
    while(val not in ['1', '2', '3']):
        print(val, 'is not a valid choice.')
        val = input('Choose an option: \n') 
    return int(val)

def main():
    val = choose_option()
    if val == 1:
        input_path = input('Type the complete path to the folder with the images: \n') 
        while not isdir(input_path):
            print('Error!', input_path, 'is not a directory.')
            input_path = input('Type the complete path to the folder with the images: \n')         
        else:            
            df_batches = create_batches(input_path)
            project_name = basename(input_path)
            move_images(input_path, df_batches, project_name)
            df_batches.to_csv(join(defaults['output_folder'], project_name, 'batches.csv'), index=None)

    elif val == 2:
        list_projects = print_projects()
        project_name = input('Type the name of the project: \n') 
        while project_name not in list_projects:
            print('Error! Project', project_name, 'does not exist.')
            project_name = input('Type the name of the project: \n')
        images_folder = join(defaults['output_folder'], project_name, defaults['images'])
        num_batches = len(listdir(images_folder))
        print('Project', project_name, 'has', num_batches, 'batches.')

        batch_start = int(input('Type the number of the first batch to be processed: \n') )
        while batch_start <= 0 or batch_start > num_batches:
            print('Error! Batch', batch_start,'does not exist.')
            batch_start = int(input('Type the number of the first batch to be processed: \n') )

        batch_end = int(input('Type the number of the last batch to be processed: \n') )
        while batch_end <= 0 or batch_end > num_batches:
            print('Error! Batch', batch_end,'does not exist.')
            batch_end = int(input('Type the number of the last batch to be processed: \n') )

        model = get_model()
        df_batches = pd.read_csv(join(defaults['output_folder'], project_name, 'batches.csv'), index_col = None)

        for i in range(batch_start, batch_end + 1):
            batch_id = 'batch_{:04d}'.format(i)
            print('Processing', batch_id)
            features, path_images = compute_features(images_folder, batch_id, model, weights_path = '')
            df = compute_projections(project_name, batch_id, features, path_images, df_batches) 
            generate_data(df, images_folder, project_name, batch_id)

if __name__ == '__main__':
   main()