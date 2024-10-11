import glob
import pandas as pd

def get_csv_files(data_dir='data') -> list:
    return glob.glob(f'{data_dir}/**/**/data.csv', recursive=True)

def create_meta_df():
    files = get_csv_files()
    dfs = []  # List to hold each DataFrame
    for f in files:
        sub_df = pd.read_csv(f)
        dfs.append(sub_df)  # Add each DataFrame to the list
    
    # Combine all DataFrames into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Save the combined DataFrame to a CSV file
    combined_df.to_csv('combined_data.csv', index=False)
    print(combined_df)

create_meta_df()