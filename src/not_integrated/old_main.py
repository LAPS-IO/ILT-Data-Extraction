"""
THIS FILE IS UNUSED

IT'S BEEN KEPT HERE AS BACKUP
"""

def main():
    val = choose_option()
    if val == 1:
        input_path = input('Type the complete path to the folder with the images: ')
        while not isdir(input_path):
            print('Error!', input_path, 'is not a directory.')
            input_path = input('Type the complete path to the folder with the images: ')
        else:
            df_batches = create_batches(input_path)
            project_name = basename(input_path)
            move_images(input_path, df_batches, project_name)
            df_batches.to_csv(join(defaults['output_folder'], project_name, 'batches.csv'), index=None)

    elif val == 2:
        project_name = choose_project()
        weights_path = input('Type the complete path to the trained model (or press Enter to load the default weights): ')

        if len(weights_path) > 0 and not exists(weights_path):
            print('Error! Model not found')
        else:
            model = get_model(load = True, num_classes = defaults['num_classes'])


        images_folder = join(defaults['output_folder'], project_name, defaults['images'])

        batch_start = choose_batch_start(project_name, images_folder)
        batch_end = choose_batch_end(images_folder, batch_start)

        df_batches = pd.read_csv(join(defaults['output_folder'], project_name, 'batches.csv'), index_col = None)

        base_id = defaults['base_tsne_id']
        features, path_images = compute_features(images_folder, base_id, model, weights_path)
        df, base_tsne = compute_projections(project_name, base_id, features, path_images, df_batches, compute_base = True, save=False)

        for i in range(batch_start, batch_end + 1):
            batch_id = 'batch_{:04d}'.format(i)
            print('Processing', batch_id)
            features, path_images = compute_features(images_folder, batch_id, model, weights_path)
            df = compute_projections(project_name, batch_id, features, path_images, df_batches, base_tsne = base_tsne)

    elif val == 3:
        project_name = choose_project()
        images_folder = join(defaults['output_folder'], project_name, defaults['images'])

        batch_start = choose_batch_start(project_name, images_folder)
        batch_end = choose_batch_end(images_folder, batch_start)
        dataframes_folder = join(defaults['output_folder'], project_name, defaults['dataframes'])

        for i in range(batch_start, batch_end + 1):
            batch_id = 'batch_{:04d}'.format(i)
            print('Processing', batch_id)

            df = pd.read_csv(join(dataframes_folder, batch_id + '_' + project_name + '.csv'), index_col=None)
            generate_data(df, images_folder, project_name, batch_id)

    elif val == 4:
        project_name = choose_project()
        images_folder = join(defaults['output_folder'], project_name, defaults['images'])
        thumbnails_folder = join(defaults['output_folder'], project_name, defaults['thumbnails'])

        batch_start = choose_batch_start(project_name, images_folder)
        batch_end = choose_batch_end(images_folder, batch_start)

        for i in range(batch_start, batch_end + 1):
            batch_id = 'batch_{:04d}'.format(i)
            print('Processing', batch_id)
            input_path = join(images_folder, batch_id, defaults['inner_folder'])
            generate_thumbnails(input_path, thumbnails_folder, batch_id, defaults['thumbnails_size'])

    elif val == 5:
        project_name = choose_project()
        images_folder = join(defaults['output_folder'], project_name, defaults['images'])

        batch_start = choose_batch_start(project_name, images_folder)
        batch_end = choose_batch_end(images_folder, batch_start)

        for i in range(batch_start, batch_end + 1):
            batch_id = 'batch_{:04d}'.format(i)
            print('Processing', batch_id)
            input_path = join(images_folder, batch_id, defaults['inner_folder'])
            add_scale(input_path, batch_id)