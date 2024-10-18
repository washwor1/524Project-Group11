import glob
import pandas as pd

def get_csv_files(data_dir='data') -> list:
    return glob.glob(f'{data_dir}/**/**/data.csv', recursive=True)
def create_meta_df():
    files = get_csv_files()
    dfs = [] 
    for f in files:
        sub_df = pd.read_csv(f)
        dfs.append(sub_df)  
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv('combined_data.csv', index=False)
    print(combined_df)
create_meta_df()