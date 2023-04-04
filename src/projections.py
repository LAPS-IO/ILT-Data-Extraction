import openTSNE
import sklearn.manifold
import numpy as np
import pandas as pd
from aux import defaults, create_dir
from os.path import join

def tsne_fit(features, n=1):
#    t = time.localtime()
#    current_time = time.strftime("%H:%M:%S", t)
#    print(current_time, 't-SNE starting...')
    
    tsne_results = sklearn.manifold.TSNE(n_components=n, learning_rate='auto', init='random', perplexity=30).fit_transform(features)
    
#    t = time.localtime()
#    current_time = time.strftime("%H:%M:%S", t)
#    print(current_time, 't-SNE computed\n')
    
    return tsne_results

def opentsne_fit(features, n=2): 
#    t = time.localtime()
#    current_time = time.strftime("%H:%M:%S", t)
#    print(current_time, 't-SNE starting...')

    tsne = openTSNE.TSNE(
        n_components=n,
        perplexity=30,
        initialization="pca",
        metric="cosine",
        random_state=0,
    )    
    tsne_results = tsne.fit(features)
    
#    t = time.localtime()
#    current_time = time.strftime("%H:%M:%S", t)
#    print(current_time, 't-SNE computed\n')
    
    return tsne_results

def opentsne_transform(features, base_tsne):
#    t = time.localtime()
#    current_time = time.strftime("%H:%M:%S", t)
#    print(current_time, 't-SNE starting...')
    
    tsne_results = base_tsne.transform(features)
    
#    t = time.localtime()
#    current_time = time.strftime("%H:%M:%S", t)
#    print(current_time, 't-SNE computed\n')
    
    return tsne_results

def compute_projections(project_name, batch_id, features, path_images, df_batches, base_tsne = None, compute_base = True):    
    print('  Computing projections...')
    if compute_base:
        base_tsne = opentsne_fit(features)
        projection = base_tsne.copy()
    else:
        projection = opentsne_transform(features, base_tsne)         
        
    path_images = np.reshape(np.array(path_images), (-1, 1))

    tsne_arr = np.hstack((path_images, projection))

    df_preds = pd.DataFrame(tsne_arr, columns =['names', 'x', 'y'])
    dataframes_folder = join(defaults['output_folder'], project_name, defaults['dataframes'])
    create_dir(dataframes_folder)

    df_filtered = df_batches[df_batches['batch'] == batch_id]
    df = pd.merge(df_preds, df_filtered, how = 'names')

    df.to_csv(join(dataframes_folder, batch_id + '_' + project_name + '.csv'), index=None)

    print('  Projections computed.')
    