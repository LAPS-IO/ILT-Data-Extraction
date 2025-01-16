import os
import sys
import tqdm


def main():
    cur_filepath = os.path.abspath(sys.argv[1])
    new_filepath = os.path.abspath(sys.argv[1][:-4] + '_D.txt')
    cur_file = open(cur_filepath, 'r')
    new_file = open(new_filepath, 'w')
    cycles = []
    for line in tqdm.tqdm(cur_file):
        t1 = line.split(' ')
        t2 = t1[0].split('_')
        t3 = t2[2].split('-')
        day = t3[2]
        month = t3[1]
        cycles.append(month + '/' + day + '/' + line)
    cycles.sort()
    new_file.writelines(cycles)
    cur_file.close()
    new_file.close()


if __name__ == '__main__':
    main()
