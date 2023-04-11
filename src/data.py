from os import listdir, mkdir
from os.path import join, exists, basename
import pandas as pd
import numpy as np
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from math import ceil, floor
from aux import defaults

#from PIL import ImageFont
#from PIL import ImageDraw 
from aux import defaults, create_dir

# generate dataframe
def create_csv(df, csv_path):        
    #df['names'] = df['names'].str.replace('png', 'jpg')
    df = df.rename(columns={'class': 'correct_label'})
    
    labels = df['correct_label'].tolist()
    df['correct_label'] = ['_' + str(l) for l in labels]

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
    df['D1'] = [0] * df.shape[0]
    df['D4'] = [0] * df.shape[0]
    df['D7'] = [0] * df.shape[0]
    
    df['thumbnails'] = df['names'].copy()
#    df['thumbnails'] = df['thumbnails'].str.replace('png', 'jpg')
    
#    df = df.rename(columns={'layer1': 'D1', 'layer2': 'D4', 'layer3': 'D7', 'layer4x': 'x', 'layer4y': 'y'})
    
    df.to_csv(csv_path, index = False)


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
    df_x = pd.to_numeric(df['x'])
    df_y = pd.to_numeric(df['y'])

    df_filtered = df[(df_x >= xrange[0]) & (df_x <= xrange[1]) & (df_y >= yrange[0]) & (df_y <= yrange[1])]
    
    x = df_filtered['x']
    y = df_filtered['y']
    names = df_filtered['names']
    
    f = plt.figure(figsize=(fig_size, fig_size), frameon=False)
    ax = plt.Axes(f, [0., 0., 1., 1.])
    ax.axis('off')
    f.add_axes(ax)
    ax.scatter(x, y, s=0) 

    for xs, ys, name in zip(x, y, names):
        path = join(images_folder, name)

        ab = AnnotationBbox(get_image(path, zoom=zoom), (xs, ys), frameon=False, box_alignment=(0, 1))
        ax.add_artist(ab)
        
    #plt.grid()
    #plt.axis('off')

    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    print(output_path)
    f.savefig(output_path, bbox_inches='tight', pad_inches = 0, dpi=100)
    
#    plt.cla()
#    plt.clf()
    plt.close(f)
    image = Image.open(output_path)
    #new_image = Image.new("RGBA", image.size, "WHITE")
    #new_image.paste(image, (0, 0), image)              
    #new_image.convert('RGB').save(output_path, "PNG") 

    return image

def add_scale(input_path, img_name):
    for img_name in listdir(input_path):
        img = cv2.imread(join(input_path, img_name))
        img_out = np.zeros([img.shape[0] + 30, img.shape[1] + 20, 3])
        img_out = 255-img_out
        
        img_out[10:img.shape[0]+10,10:img.shape[1]+10] = img
        units = defaults['pixel_size'] * img.shape[1] * defaults['ruler_ratio']
        units = int(round(units/100))
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = 100*units
        if size > 900:
            size = 1000
        ruler_size = int(round(size/defaults['pixel_size']))

        img_out = Image.fromarray(np.uint8(img_out)).convert('RGB')
        draw = ImageDraw.Draw(img_out)

        if size < 1000:
            text = str(size) + ' µm'
        else:
            text = str(int(size/1000)) + ' mm'
        draw.text((12, img.shape[0] + 12), text,(0, 0, 0))
        
        shape = [(10, img.shape[0] + 25), (ruler_size + 10, img.shape[0] + 25)]
        draw.line(shape, fill = 'black', width = 1)

        shape = [(10, img.shape[0] + 23), (10, img.shape[0] + 27)]
        draw.line(shape, fill = 'black', width = 1)

        shape = [(ruler_size + 10, img.shape[0] + 23), (ruler_size + 10, img.shape[0] + 27)]
        draw.line(shape, fill = 'black', width = 1)

        img_out.save(join(input_path, img_name))

## rescale image for thumbnails
def generate_thumbnails(input_path, thumbnails_folder, batch_id, max_size):
    create_dir(thumbnails_folder)

    thumbnails_path = join(thumbnails_folder, batch_id)
    create_dir(thumbnails_path)
    
    inner_path = join(thumbnails_path, defaults['inner_folder'])
    create_dir(inner_path)

    for img_name in listdir(input_path):
        img = cv2.imread(join(input_path, img_name))
        if img.shape[0] > max_size:
            scale = img.shape[0]/max_size
            dims = (int(img.shape[1]/scale),int(img.shape[0]/scale))
            resized = cv2.resize(img, dims, interpolation = cv2.INTER_AREA)
        else:
            resized = img
#        output_name = join(inner_path, img_name[:-4] + '.jpg')
        output_name = join(inner_path, img_name)
        cv2.imwrite(output_name, resized)


def generate_data(df, images_folder, project_name, batch_id, range = 100):
    print('Generating background...')
    #match = 'last_after_p2'

#    dataset_files = listdir(dataset_folder)

    dataframes_folder = join(defaults['output_folder'], project_name, defaults['dataframes'])
    create_dir(dataframes_folder)
    print(dataframes_folder)

    backgrounds_folder = join(defaults['output_folder'], project_name, defaults['backgrounds'])
    create_dir(backgrounds_folder)
    print(backgrounds_folder)
        
    fig_size = 40
    factor = defaults['map_factor'] #default 2 tsne, 20 umap
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
    
    images_folder_batch = join(images_folder, batch_id, defaults['inner_folder'])
    image = map_of_images(df, xrange, yrange, images_folder_batch, backgrounds_path, zoom, fig_size)    
    
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
    
    csv_path = join(dataframes_folder, batch_id + '_' + project_name + '.csv')
    create_csv(df, csv_path)
    
