import os

defaults = {
    'BATCH_MAX_SIZE': 8000,
    'output': 'output',
    'images': 'images',
    'dataframes': 'dataframes',
    'backgrounds': 'backgrounds',
    'thumbnails': 'thumbnails',
    'root': os.path.dirname(os.getcwd()),
    'output_folder': '',
    'inner_folder': 'samples',
    'thumbnails_size': 100,
    'pixel_size': 12.87,
    'ruler_ratio': 0.5,
    'base_tsne_id': 'batch_0001',
    'num_classes': 25,
    'map_factor': 2 #decrease to increase the size of the imagens in the backgrounds
}

def update_defaults():
  defaults['output_folder'] = os.path.join(defaults['root'], defaults['output'])

update_defaults()


# Input:
  # (1) path: a string containing a path
  # (2) ignore: used to decide what to do when the path already exists
# Output:
  # (1) returns True if the path was created succesfully
  #     of if ignore was set to True. Returns False if
  #     both the path existed and ignore was set to False
def create_dir(path, ignore=True):
    if os.path.exists(path):
        return ignore
    else:
        os.mkdir(path)
        return True
