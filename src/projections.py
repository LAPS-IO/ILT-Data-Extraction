import openTSNE
import sklearn.manifold
import numpy as np
import pandas as pd
import os
from aux import defaults

def tsne_fit(features, n=1):
    # t = time.localtime()
    # current_time = time.strftime("%H:%M:%S", t)
    # print(current_time, 't-SNE starting...')

    tsne_results = sklearn.manifold.TSNE(n_components=n, learning_rate='auto', init='random', perplexity=30).fit_transform(features)

    # t = time.localtime()
    # current_time = time.strftime("%H:%M:%S", t)
    # print(current_time, 't-SNE computed\n')

    return tsne_results

def opentsne_fit(features, n=2):
    # t = time.localtime()
    # current_time = time.strftime("%H:%M:%S", t)
    # print(current_time, 't-SNE starting...')

    tsne = openTSNE.TSNE(
        n_components=n,
        perplexity=30,
        initialization="pca",
        metric="cosine",
        random_state=0,
    )
    tsne_results = tsne.fit(features)

    # t = time.localtime()
    # current_time = time.strftime("%H:%M:%S", t)
    # print(current_time, 't-SNE computed\n')

    return tsne_results

def opentsne_transform(features, base_tsne):
    # t = time.localtime()
    # current_time = time.strftime("%H:%M:%S", t)
    # print(current_time, 't-SNE starting...')

    tsne_results = base_tsne.transform(features)

    # t = time.localtime()
    # current_time = time.strftime("%H:%M:%S", t)
    # print(current_time, 't-SNE computed\n')

    return tsne_results

def compute_projections(output_path, project_name, batch_id, features, path_images, df_batches, predictions, base_tsne=None, compute_base=True, save=True):
    if compute_base:
        base_tsne = opentsne_fit(features)
        projection = base_tsne.copy()
    else:
        projection = opentsne_transform(features, base_tsne)

    path_images = np.reshape(np.array(path_images), (-1, 1))
    predictions_arr = np.reshape(np.array(predictions), (-1, 1))

    tsne_arr = np.hstack((path_images, projection, predictions_arr))

    df_preds = pd.DataFrame(tsne_arr, columns =['names', 'x', 'y', 'pred'])
    dataframes_folder = os.path.join(output_path, defaults['dataframes'])
    if not os.path.isdir(dataframes_folder):
        os.mkdir(dataframes_folder, mode=0o755)

    df_filtered = df_batches[df_batches['batch'] == batch_id]
    df = pd.merge(df_preds, df_filtered, on='names')

    if save:
        df.to_csv(os.path.join(dataframes_folder, batch_id + '_' + project_name + '.csv'), index=None)

    return df, base_tsne