import json
import multiprocessing as mp
import os
import timeit
from datetime import timedelta

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import tqdm
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from aux import defaults


def update(progress_bar):
    progress_bar.update(1)


# generate dataframe
def create_csv(df, csv_path):
    df = df.rename(columns={'klass': 'correct_label'})
    labels = df['correct_label'].tolist()
    df['correct_label'] = ['_' + str(l) for l in labels]

    classes = df['correct_label'].unique()
    classes.sort()
    map_color = {}
    for i in range(len(classes)):
        map_color[classes[i]] = i + 1

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

    df.to_csv(csv_path, index=False)


# generate backgrounds
def get_image(path, paint=False, color=(1, 1, 1), zoom=0.2, dim=255):
    img = PIL.Image.open(path).convert('RGBA')
    img = np.array(img)
    if paint:
        img[:,:,0] = np.uint8(img[:,:,0] * color[0])
        img[:,:,1] = np.uint8(img[:,:,1] * color[1])
        img[:,:,2] = np.uint8(img[:,:,2] * color[2])
        img[:,:,3] = dim
    img = PIL.Image.fromarray(img)

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
        path = os.path.join(images_folder, name)
        ab = AnnotationBbox(get_image(path, zoom=zoom), (xs, ys), frameon=False, box_alignment=(0, 1))
        ax.add_artist(ab)

    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    f.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(f)


def generate_bkg(backgrounds_folder, df_folder, images_folder, project_name, batch_num, range=100):
    batch_id = 'batch_{:04d}'.format(batch_num)
    df = pd.read_csv(os.path.join(df_folder, batch_id + '_' + project_name + '.csv'), index_col=None)

    fig_size = 40
    factor = defaults['map_factor'] # defaults: 2 tsne, 20 umap
    xrange = [-range, range]
    yrange = [-range, range]
    zoom = fig_size / (factor * (xrange[1] - xrange[0]))

    backgrounds_path = os.path.join(backgrounds_folder, batch_id + '_' + project_name + '.png')
    images_folder_batch = os.path.join(images_folder, batch_id, defaults['inner_folder'])
    map_of_images(df, xrange, yrange, images_folder_batch, backgrounds_path, zoom, fig_size)

    csv_path = os.path.join(df_folder, batch_id + '_' + project_name + '.csv')
    create_csv(df, csv_path)

def read_labels(label_path):
    json_file = open(label_path)
    labels_d = json.load(json_file)
    
    return labels_d

def label_predictions(batch_df, label_path):
    labels_d = read_labels(label_path)
    labels_d = {v: k for k, v in labels_d.items()}
    batch_df['colors'] = batch_df['pred'] + 1
    batch_df['pred'] = batch_df['pred'].replace(labels_d)
    batch_df['correct_label'] = batch_df['pred']
    batch_df['manual_label'] = batch_df['pred']

def clean_merge_dfs(df1, df2):
    df2.drop(['x', 'y', 'correct_label', 'manual_label', 'colors',
              'custom_data', 'x2', 'y2', 'x3', 'y3',
              'D1', 'D4', 'D7', 'thumbnails'], axis=1, inplace=True)
    return pd.concat([df1, df2])
