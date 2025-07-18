import os

import numpy as np
import openTSNE
import pandas as pd
import sklearn.manifold
from PIL import ImageFile

from aux import defaults


def tsne_fit(features, n=1):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    tsne = sklearn.manifold.TSNE(n_components=n, learning_rate='auto', init='random', perplexity=30)
    tsne_results = tsne.fit_transform(features)
    return tsne_results

def opentsne_fit(features, n=2):
    tsne = openTSNE.TSNE(
        n_components=n,
        perplexity=30,
        initialization="pca",
        metric="cosine",
        random_state=0,
    )
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    tsne_results = tsne.fit(features)
    return tsne_results

def opentsne_transform(features, base_tsne):
    tsne_results = base_tsne.transform(features)
    return tsne_results

def compute_projections(output_path, project_name, batch_id, features, path_images, df_batches, predictions, confs, base_tsne=None, compute_base=False, save=True):
    if compute_base:
        base_tsne = opentsne_fit(features)
        projection = base_tsne.copy()
    else:
        projection = opentsne_transform(features, base_tsne)

    path_images = np.reshape(np.array(path_images), (-1, 1))
    predictions_arr = np.reshape(np.array(predictions), (-1, 1))
    tsne_arr = np.hstack((path_images, projection, predictions_arr))
    df_preds = pd.DataFrame(tsne_arr, columns =['names', 'x', 'y', 'pred'])
    
    preds1 = []
    confs1 = []
    preds2 = []
    confs2 = []
    preds3 = []
    confs3 = []

    for conf in confs:
        preds1.append(conf['top1'][0])
        confs1.append(conf['top1'][1])
        preds2.append(conf['top2'][0])
        confs2.append(conf['top2'][1])
        preds3.append(conf['top3'][0])
        confs3.append(conf['top3'][1])

    df_preds['preds1'] = preds1
    df_preds['confs1'] = confs1
    df_preds['preds2'] = preds2
    df_preds['confs2'] = confs2
    df_preds['preds3'] = preds3
    df_preds['confs3'] = confs3

    df_filtered = df_batches[df_batches['batch'] == batch_id]
    df = pd.merge(df_preds, df_filtered, on='names')

    if save:
        dataframes_folder = os.path.join(output_path, defaults['dataframes'])
        if not os.path.isdir(dataframes_folder):
            os.mkdir(dataframes_folder, mode=0o755)
        df.to_csv(os.path.join(dataframes_folder, f'{batch_id}_{project_name}.csv'), index=None)

    return base_tsne