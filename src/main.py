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
    output_path = defaults['output']
    projects = [f for f in listdir(output_path) if isdir(join(output_path, f))]
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
            images_folder = move_images(input_path, df_batches, dataset_name)
    elif val == 2:
        list_projects = print_projects()
        project_name = input('Type the name of the project: \n') 
        while project_name not in list_projects:
            print('Error! Project', project_name, 'does not exist.')
            input_path = input('Type the name of the project: \n')
        input_path = join(defaults['output'], project_name) 
        compute_features(input_path, project_name, weights_path = '')

if __name__ == '__main__':
   main()