import os
import os.path
import sys

import numpy as np
import tqdm
from PIL import Image


def remove_scale(img_path):
    try:
        img = Image.open(img_path)
    except:
        print('Error processing:', img_path)
        os.remove(img_path)
        return False
    else:
        arr = np.array(img)
        w, h = img.size
        clean = False
        passes = 0
        while(not clean):
            avg1 = np.mean(arr[0:10, 0:w])
            avg2 = np.mean(arr[10:h-10, 0:10])
            avg3 = np.mean(arr[10:h-10, w-10:w])
            avg = (avg1 + avg2 + avg3) / 3
            if avg > 240:
                img = img.crop((10, 10, w-10, h-20))
                passes += 1
            else:
                clean = True
        img.save(img_path)
        return passes


def main():
    input_path = os.path.abspath(sys.argv[1])

    print('Generating image list for:', input_path)
    imgs = []
    for pwd, children, files in os.walk(input_path):
        rel_pwd = pwd[len(input_path) + 1:]
        imgs += [ os.path.join(pwd, file) for file in files if (file.endswith('.png') or file.endswith('.jpg')) ]
    print('Found', len(imgs), "images.\n")

    passes = {}
    for img_path in tqdm.tqdm(imgs, unit='img', ascii=True, ncols=79):
        value = remove_scale(img_path)
        if value in passes:
            passes[value] += 1
        else:
            passes[value] = 1
    print(passes)
    print('Finished.')

if __name__ == '__main__':
    main()
