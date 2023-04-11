import pandas as pd
from os import listdir
from sys import argv
from os.path import join

features_columns = ['names', 'Image State (T/F)', 'SegmentationMethod', 'Area (pxl)',
           'Image Width (pxl)', 'Image Size (pxl)', 'circularity', 'Elongation',
           'Rectangularity', 'Mean Intensity', 'Median Intensity', 'Contrast',
           'Solidity', 'Percent_on_border', 'Validation State', 'Number Holes', 'Event Indexer',
           'Class Name', 'ClassName_previous', 'Background Image']


def main():
    path = argv[1]
    lif_path = argv[2]
    
    df_lif = pd.read_csv(lif_path, index_col = None, names = features_columns, sep = ' ')
    
    for f in listdir(path):
        filename = join(path, f)
        df = pd.read_csv(filename, index_col = None)
        df = df.merge(df_lif, how = 'inner', on = ['names'])
        print(df)
            
        df.to_csv(filename, index=None)


if __name__ == '__main__':
    main()
