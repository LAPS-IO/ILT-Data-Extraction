import sys
import os
import random
import tqdm

def main():
    path, tr_n, te_n, va_n = sys.argv[1:]
    main_path = os.path.abspath(path)
    folder_sizes = {'train': int(tr_n), 'test': int(te_n), 'valid': int(va_n)}
    for folder in folder_sizes:
        cwd = os.path.join(main_path, folder)
        klasses = [os.path.basename(f.path) for f in os.scandir(cwd) if f.is_dir()]
        for klass in klasses:
            cwd2 = os.path.join(cwd, klass)
            files = [os.path.join(cwd2, f) for f in os.listdir(cwd2)]
            if len(files) > folder_sizes[folder]:
                print('cleaning: ' + cwd2)
                random.shuffle(files)
                to_del = files[folder_sizes[folder]:]
                for file in tqdm.tqdm(to_del, unit='img'):
                    os.remove(file)


if __name__ == '__main__':
    main()