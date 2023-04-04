from os import listdir, mkdir
from os.path import join, exists, basename
import pandas as pd
import numpy as np
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from math import ceil, floor

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from aux import defaults, create_dir

# generate dataframe
def create_csv(df, dataframes_path, filename):        
    df['names'] = df['names'].str.replace('png', 'jpg')
    
    labels = df['correct_label'].tolist()
    df['correct_label'] = ['_' + l for l in labels]

    classes = df['correct_label'].unique()
    classes.sort()
    map_color = {}
    for i in range(len(classes)):
        map_color[classes[i]] = i+1

    
    df['manual_label'] = df['correct_label']
    list_colors = [map_color[c] for c in df['manual_label'].tolist()]
    df['colors'] = list_colors
    df['custom_data'] = [i for i in range(df.shape[0])]
    df['x2'] = [0] * df.shape[0]
    df['y2'] = [0] * df.shape[0]
    df['x3'] = [0] * df.shape[0]
    df['y3'] = [0] * df.shape[0]
    df['thumbnails'] = df['names']
    
    df = df.rename(columns={'layer1': 'D1', 'layer2': 'D4', 'layer3': 'D7', 'layer4x': 'x', 'layer4y': 'y'})
    
    df.to_csv(join(dataframes_path, filename + '.csv'), index = False)


## generate backgrounds
def get_image(path, paint = False, color = (1, 1, 1), zoom=0.2, dim = 255):
    img = Image.open(path).convert('RGBA')
    img = np.array(img)
    if paint:
        img[:,:,0] = np.uint8(img[:,:,0] * color[0])
        img[:,:,1] = np.uint8(img[:,:,1] * color[1])
        img[:,:,2] = np.uint8(img[:,:,2] * color[2])
        img[:,:,3] = dim
    #img = img[10:img.shape[0]-20, 10:img.shape[1]-10, :]
    img = Image.fromarray(img)
    
    return OffsetImage(img, zoom=zoom)

def map_of_images(df, xrange, yrange, images_folder, output_path, zoom, fig_size=40):
    df_filtered = df[(df['layer4x'] >= xrange[0]) & (df['layer4x'] <= xrange[1]) & (df['layer4y'] >= yrange[0]) & (df['layer4y'] <= yrange[1])]
    
    x = df_filtered['layer4x']
    y = df_filtered['layer4y']
    names = df_filtered['names']
    classes = df_filtered['Class']
    
    f = plt.figure(figsize=(fig_size, fig_size), frameon=False)
    ax = plt.Axes(f, [0., 0., 1., 1.])
    ax.axis('off')
    f.add_axes(ax)
    ax.scatter(x, y, s=0) 

    for xs, ys, c, name in zip(x, y, classes, names):
        path = join(images_folder, c, name)

        ab = AnnotationBbox(get_image(path, zoom=zoom), (xs, ys), frameon=False, box_alignment=(0, 1))
        ax.add_artist(ab)
        
    #plt.grid()
    #plt.axis('off')

    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    f.savefig(output_path, bbox_inches='tight', pad_inches = 0, dpi=100)
    
    plt.cla()
    plt.clf()
    image = Image.open(output_path)
    #new_image = Image.new("RGBA", image.size, "WHITE")
    #new_image.paste(image, (0, 0), image)              
    #new_image.convert('RGB').save(output_path, "PNG") 

    return image

## rescale image for thumbnails
def rescale(input_path, output_path, img_name):
    img = cv2.imread(join(input_path, img_name))
    if img.shape[0] > 100:
        scale = img.shape[0]/100
        dims = (int(img.shape[1]/scale),int(img.shape[0]/scale))
        resized = cv2.resize(img, dims, interpolation = cv2.INTER_AREA)
    else:
        resized = img
    output_name = join(output_path, img_name[:-4] + '.jpg')
    cv2.imwrite(output_name, resized)


def generate_data(df, images_folder, project_name, batch_id, range = 80):
    print('Generating background...')
    #match = 'last_after_p2'

#    dataset_files = listdir(dataset_folder)

    dataframes_folder = join(defaults['output'], project_name, defaults['dataframes'])
    create_dir(dataframes_folder)

    backgrounds_folder = join(defaults['output'], project_name, defaults['backgrounds'])
    create_dir(backgrounds_folder)
        
    fig_size = 40
    factor = 2 #default 2 tsne, 20 umap
    xrange = [-range, range]
    yrange = [-range, range]
    zoom = fig_size/(factor*(xrange[1]-xrange[0])) 

#    df_map = pd.read_csv(join(dataset_folder, 'map_images.csv'))

    #batch_name = 'batch_{:04d}'.format(batch_id)
    #map_batch = df_map[df_map['Batch'] == batch_name]
    #map_batch = map_batch.rename(columns={'Image': 'names'})
    
    # Dataframe generation
    #predictions_path = join(predictions_folder, batch_name + '_preds.csv')
    #df = pd.read_csv(predictions_path, index_col = None)
    #df = pd.merge(df, map_batch, on=['names'])
    #df = remove_ppms(df)
    
    # Background generation
    backgrounds_path = join(backgrounds_folder, batch_id + '_' + project_name + '.png')
    
    images_folder_batch = join(images_folder, batch_id)
    image = map_of_images(df, xrange, yrange, images_folder_batch, backgrounds_path, zoom, fig_size)    

    current_time = datetime.now().strftime("%H:%M:%S")
    print('    (', current_time, ') - background computed: ', df['layer4x'].min(), df['layer4x'].max(), df['layer4y'].min(), df['layer4y'].max(), image.size[0], image.size[1])
    
    # Scale + Thumbnail generation 
    #thumbnails_path = join(thumbnails_folder, batch_name)
    #if not exists(thumbnails_path):
    #    mkdir(thumbnails_path)
        
    #thumbnails_samples_path = join(thumbnails_path, 'samples')
    #if not exists(thumbnails_samples_path):
    #    mkdir(thumbnails_samples_path)

#    scales_path = join(images_folder, batch_name)
#    if not exists(scales_path):
#        mkdir(scales_path)
        
#    scales_samples_path = join(scales_path, 'samples')
#    if not exists(scales_samples_path):
#        mkdir(scales_samples_path)
    
    #for index, row in df.iterrows():
    #    img = row['names']
    #    class_path = join(input_images_folder, row['Class'])
    #    rescale(class_path, thumbnails_samples_path, img)
    #    add_scale(class_path, scales_samples_path, img)
    
    current_time = datetime.now().strftime("%H:%M:%S")
    print('    (', current_time, ') - images + thumbnails computed')
    
    create_csv(df, dataframes_folder, filename)
    
    current_time = datetime.now().strftime("%H:%M:%S")
    print('    (', current_time, ') - CSV computed')