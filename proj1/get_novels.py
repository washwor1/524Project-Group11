'''
    Data Retrieval Scripts
    
    Downloads the raw data files from Project Gutenberg locally
'''

import requests
import os
import pandas as pd
import glob

def process_row(output_dir, row):
    """
    Download the book associated with each row of sources.csv
    """
    req = requests.get(row['url'], allow_redirects=True)
    author_name = row['author'].lower().replace(' ', '_').replace('.', '')
    path = f"{output_dir}/{author_name}"
    name = row['title'].lower().replace(' ', '_').replace(':', '_').replace(',','')
    os.makedirs(path, exist_ok=True)
    
    with open(f"{path}/raw_{name}.txt", 'wb') as fp:
        fp.write(req.content)
        print(f"Retrieved \'{row['title']}\'")


def get_novels(data_dir = 'data'):
    '''
    Wrapper function to download the books from the db

    returns the path that the books were saved to

    If there is an error, it will return None
    '''
    try:
        t = glob.glob(f"{data_dir}/**/raw*.txt")
        if len(t) != 0:
            print("get_novels: skipping, as downloaded files have been found")
        else:
            df = pd.read_csv(f"{data_dir}/sources.csv")
            df.apply(lambda x: process_row(data_dir, x), axis=1)
            print(f"get_novels: Downloaded {df.shape[0]} books")
    except: 
        print("get_novels: Error downloading novel data.")
        raise

if __name__ == "__main__":
    get_novels()