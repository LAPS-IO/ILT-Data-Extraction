import os
import sys


def list_corrupt(dirpath):
    for path, dirs, files in os.walk(dirpath):
        dirs.sort()
        corrupt = False
        dl = dirs.copy()
        for d in dl:
            if d[:6] == 'Basler':
                print(d)
        for f in files:
            filepath = os.path.join(path, f)
            if os.stat(filepath).st_size == 0:
                corrupt = True
                continue
        if corrupt:
            print(path)


def main():
    list_corrupt(os.path.abspath(sys.argv[1]))


if __name__ == '__main__':
    main()
