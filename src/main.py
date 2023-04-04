import sys
from batch import create_batches, move_images
from features import compute_features 
from os.path import basename, isdir

def print_choices():
    print('1: Create batches')
    print('2: Extract features')
    print('3: Compute backgrounds and dataframes')

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
            print(df_batches)
            dataset_name = basename(input_path)
            print(dataset_name)
            images_folder = move_images(input_path, df_batches, dataset_name)
    elif val == 2:
        compute_features(images_folder, project_name = dataset_name, weights_path = '')

if __name__ == '__main__':
   main()