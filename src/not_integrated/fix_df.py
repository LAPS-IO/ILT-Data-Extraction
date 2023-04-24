import sys
import os
import tqdm
import pandas as pd
#from parallel_pandas import ParallelPandas

def main():
    # initialize parallel-pandas
    # ParallelPandas.initialize(n_cpu=32, split_factor=8, disable_pr_bar=True)
    df_path = sys.argv[1]
    for csv in tqdm.tqdm(os.listdir(df_path), ascii=True):
        full_path = os.path.join(df_path, csv)
        df = pd.read_csv(full_path)
        df['correct_label'] = df['pred']
        df['manual_label'] = df['pred']
        df.to_csv(full_path)


if __name__ == "__main__":
    main()
