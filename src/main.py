import sys
from batch import create_batches, move_images
from aux import defaults
from features import compute_features 
from os import listdir
from os.path import basename, isdir, join

def print_choices():
    print('1: Create batches')
    print('2: Extract features')
    print('3: Compute backgrounds and dataframes')

def print_projects():
    output_path = defaults['output_path']
    projects = [f for f in listdir(output_path) if isdir(join(output_path, f))]
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
            dataset_name = basename(input_path)
            move_images(input_path, df_batches, dataset_name)
    elif val == 2:
        list_projects = print_projects()
        project_name = input('Type the name of the project: \n') 
        while project_name not in list_projects:
            print('Error! Project', project_name, 'does not exist.')
            project_name = input('Type the name of the project: \n')
        input_path = join(defaults['output_path'], project_name)
        images_path = join(input_path, defaults['images'])
        num_batches = len(listdir(input_path))
        print('Project', project_name, 'has', num_batches, 'batches.')

        batch_start = int(input('Type the number of the first batch to be processed: \n') )
        while batch_start <= 0 or batch_start > num_batches:
            print('Error! Batch', batch_start,'does not exist.')
            batch_start = int(input('Type the number of the first batch to be processed: \n') )

        batch_end = int(input('Type the number of the last batch to be processed: \n') )
        while batch_end <= 0 or batch_end > num_batches:
            print('Error! Batch', batch_end,'does not exist.')
            batch_end = int(input('Type the number of the last batch to be processed: \n') )

        compute_features(images_path, batch_start, batch_end, weights_path = '')

if __name__ == '__main__':
   main()