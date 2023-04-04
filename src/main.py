import sys
from batch import create_batches, move_images
from features import compute_features 
from os.path import basename

def main():
    input_path = sys.argv[1]

    df_batches = create_batches(input_path)
    dataset_name = basename(input_path)
    images_folder = move_images(input_path, df_batches, dataset_name)
    if len(images_folder) > 0: #images moved succesfully
        compute_features(images_folder, project_name = dataset_name, weights_path = '')

if __name__ == '__main__':
   main()