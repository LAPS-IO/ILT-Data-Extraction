import pandas as pd
import tqdm
import sys
import os


def main():
    bp = os.path.abspath(sys.argv[1])
    dfb = pd.read_csv(os.path.join(bp, 'batches.csv'))
    dfc = pd.read_csv(os.path.join(bp, f'{os.path.basename(bp)[2:]}.csv'))
    dff = pd.merge(dfb, dfc, how='inner', on=['names'])
    kls = ['sombra', 'detritos', 'fitoplancton', 'zooplancton', 'multiplos']
    for k in kls:
        dfk = dff[dff['pred'] == k]
        fi = open(os.path.join(bp, f'{os.path.basename(bp)[2:]}_{k}.txt'), 'w')
        for r in tqdm.tqdm(dfk.itertuples()):
            cp = r[2]
            fn = r[1]
            li = f'{cp}/{fn}\n'
            fi.write(li)
        fi.close()


if __name__ == '__main__':
    main()
