from PIL import Image
from os import listdir, getcwd, mkdir
from os.path import join, isdir, exists
from sys import argv
import numpy as np

def main():
    input_folder = argv[1]

    count = 0

    folders = listdir(input_folder)
    folders.sort() 

    for c in folders:
        count = 0
        class_path = join(input_folder, c)
        if isdir(class_path):
            inner_path = join(class_path, 'samples')
            for f in listdir(inner_path):
                img_path = join(inner_path, f)
                im = Image.open(img_path)
                arr = np.array(im)
                w, h = im.size
                avg1 = np.mean(arr[0:10,0:w]) 
                avg2 = np.mean(arr[10:h-10,0:10]) 
                avg3 = np.mean(arr[10:h-10,w-10:w]) 
                avg = (avg1 + avg2 + avg3)/3
                if avg > 240:                
                    resized = im.crop((10, 10, w-10, h-20))
                    resized.save(img_path)
                    count += 1
                #else:
                    #print(f, avg)
        print('Updated ' + str(count) + ' images from ' + c)


    print('Finished.')
    
if __name__ == '__main__':
    main()
