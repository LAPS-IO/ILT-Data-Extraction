import pandas as pd
import sys
import os


def main():
    # get csv paths
    csv1, csv2 = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    # read csvs
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)
    df3 = pd.merge(df1, df2, how='inner', on=['names'])
    df4 = df2[~df2['names'].isin(df1['names'])]
    basepath = '/raid/Salvador_raw_imgs_frames/LPD_new'
    print(df3)
    df4.to_csv(os.path.join(basepath,
                            f'N_{sys.argv[1][-11:-4]}',
                            f'{sys.argv[1][-11:]}'), index=False)


if __name__ == '__main__':
    main()
