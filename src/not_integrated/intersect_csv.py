import sys
import pandas as pd


def main():
    d1 = pd.read_csv(sys.argv[1])
    d2 = pd.read_csv(sys.argv[2])
    d3 = pd.merge(d1, d2, how='inner', on=['names', 'pred'])
    if d3.empty:
        print('Empty OK')
        d4 = pd.concat([d1, d2])
        assert d4.shape[0] == d1.shape[0] + d2.shape[0]
        d4.to_csv(sys.argv[2][:-4] + 'N.csv', index=False)


if __name__ == '__main__':
    main()
