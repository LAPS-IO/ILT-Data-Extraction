import os
import cv2
import sys
import tqdm

def rotate3(img, filename):
    for i in range(3):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(f'{filename[:-4]}_rot{i}{filename[-4:]}', img)

def flip2(img, filename):
    flip_opt = {0: 'x', 1: 'y'}
    for i in flip_opt:
        flip = cv2.flip(img, i)
        cv2.imwrite(f'{filename[:-4]}_flip_{flip_opt[i]}{filename[-4:]}', flip)

def main():
    input_path = os.path.abspath(sys.argv[1])

    print('Generating image list for:', input_path)
    imgs = []
    for pwd, children, files in os.walk(input_path):
        imgs += [ os.path.join(pwd, file) for file in files if (file.endswith('.png') or file.endswith('.jpg')) ]
    print('Found', len(imgs), "images.\n")
    for img_name in tqdm.tqdm(imgs, unit='img'):
        img = cv2.imread(img_name)
        rotate3(img, img_name)
        flip2(img, img_name)


if __name__ == '__main__':
    main()