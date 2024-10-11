import requests
import os
import pandas as pd

def process_row(output_dir, row):
    """
    Download the book associated with each row of sources.csv
    """
    req = requests.get(row['url'], allow_redirects=True)
    print(req)
    author_name = row['author'].lower().replace(' ', '_').replace('.', '')
    path = f"{output_dir}/{author_name}"
    name = row['title'].lower().replace(' ', '_').replace(':', '_').replace(',','')
    print(name)
    os.makedirs(path, exist_ok=True)
    
    with open(f"{path}/raw_{name}.txt", 'wb') as fp:
        fp.write(req.content)
        print(f"Retrieved \'{row['title']}\'")


def get_novels_wrapper(output_folder_name = 'data'):
    '''
    Wrapper function to download the books from the db

    returns the path that the books were saved to

    If there is an error, it will return None
    '''
    try:
        df = pd.read_csv("sources.csv")
        df.apply(lambda x: process_row(output_folder_name, x), axis=1)
        print(f"get_novels_wrapper: Downloaded {df.shape[0]} books")
        return output_folder_name
    except: 
        print("get_novels_wrapper: Error downloading novel data.")
        return None