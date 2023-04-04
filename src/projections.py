import openTSNE
import sklearn.manifold


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

def compute_projections():
        base_tsne = None

    batch_ids = []
    for batch_id in range(start_batch, end_batch):
        batch_ids.append(batch_id)
        
    for batch_id in batch_ids:
        print('------- Batch ' + str(batch_id) + ' -------')
        activation = {}
        
        df_batch = df[df['Batch'] == 'batch_{:04d}'.format(batch_id)]
        
        test_list = []
        
        for index, row in df_batch.iterrows():
            test_list.append(join(base_path, 'batches_all', 'batch_{:04d}'.format(batch_id), row['Image']))

        test_data = IODataset(test_list, transform=test_transform)
        test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=False)

        images_path = []
        predictions = []
        features1 = None
        features2 = None
        features3 = None
        features4 = None

        with torch.no_grad():
            for data, paths in tqdm(test_loader):
                data = data.to(device)
                
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                    output = model(data)
                    if features1 is None:
                        features1 = torch.amax(activation['layer1'], (1, 2))
                        features2 = torch.amax(activation['layer2'], (1, 2))
                        features3 = torch.amax(activation['layer3'], (1, 2))
                        features4 = torch.amax(activation['layer4'], (2, 3))
                    else:
                        aux1 = torch.amax(activation['layer1'], (1, 2))
                        aux2 = torch.amax(activation['layer2'], (1, 2))
                        aux3 = torch.amax(activation['layer3'], (1, 2))
                        aux4 = torch.amax(activation['layer4'], (2, 3))
                        features1 = torch.vstack((features1, aux1))
                        features2 = torch.vstack((features2, aux2))
                        features3 = torch.vstack((features3, aux3))
                        features4 = torch.vstack((features4, aux4))
                    
                paths = list(paths)

                preds = output.argmax(dim=1)
                preds_list = []
                for i in range(preds.shape[0]):
                    preds_list.append(map_label[preds[i].item()])
                    paths[i] = basename(paths[i])
                predictions.extend(preds_list)
                images_path.extend(paths)
        
        features1 = features1.cpu().detach().numpy()
        features2 = features2.cpu().detach().numpy()
        features3 = features3.cpu().detach().numpy()
        features4 = features4.cpu().detach().numpy()
        
        arr_files = np.array(images_path).reshape(len(images_path), -1)
        arr = np.hstack([arr_files, features4])
        cs = ['names']
        for i in range(features4.shape[1]):
            cs.append('f_' + str(i+1))
        df_features = pd.DataFrame(arr, columns = cs)
        df_features

        df_features.to_csv('features.csv', index=None)

        tsne1 = tsne_fit(features1, n = 1)
        tsne2 = tsne_fit(features2, n = 1)
        tsne3 = tsne_fit(features3, n = 1)
        
        if batch_id == 9999:
            base_tsne = opentsne_fit(features4)
            tsne4 = base_tsne.copy()
        else:
            tsne4 = opentsne_transform(features4, base_tsne)         
            
        images_path = np.reshape(np.array(images_path), (-1, 1))
        predictions = np.reshape(np.array(predictions), (-1, 1))
        
        tsne_preds = np.hstack((images_path, tsne1, tsne2, tsne3, tsne4, predictions))
        print(tsne_preds.shape)
        
        df_preds = pd.DataFrame(tsne_preds, columns =['names', 'layer1', 'layer2', 'layer3', 'layer4x', 'layer4y', 'correct_label'])
        df_preds.to_csv(join(output_path, 'batch_{:04d}_preds.csv'.format(batch_id)), index=None)
            
        cur = timer()
        time_diff = cur - start
        print('batch', batch_id, time_diff/60)