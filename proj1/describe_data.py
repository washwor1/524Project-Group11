import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

def get_csv_files(data_dir='data') -> str:
    return glob.glob(f'{data_dir}/**/**/data.csv')

def create_meta_df():
    files = get_csv_files()
    dfs = []
    for f in files:
        l = f.split('\\')[1:]
        sub_df = pd.read_csv(f)
        sub_df['author'] = l[0]
        sub_df['book_name'] = l[1]
        sub_df.drop(columns=['text'], inplace=True)
        dfs.append(sub_df)
    df = pd.concat(dfs)
    print(df)
create_meta_df()