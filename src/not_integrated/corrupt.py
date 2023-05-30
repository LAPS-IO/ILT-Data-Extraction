import os
import sys

import tqdm


def main(input_path):
    print('Looking for corrupt images in', input_path)
    imgs = []
    for pwd, children, files in os.walk(input_path):
        rel_pwd = pwd[len(input_path) + 1:]
        imgs += [ (file, rel_pwd) for file in files if (file.endswith('.png') or file.endswith('.jpg')) ]

    corrupt_list = open("%s.txt" % (os.path.basename(input_path)), 'w')
    corrupt_imgs = 0
    for img in tqdm.tqdm(imgs, unit='img'):
        if os.stat(img).st_size == 0:
            corrupt_imgs += 1
            corrupt_list.write(img)
    print('Found', corrupt_imgs, 'corrupt images!')


if __name__ == '__main__':
    main(os.path.abspath(sys.argv[1]))