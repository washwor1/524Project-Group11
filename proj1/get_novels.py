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
    name = row['title'].lower().replace(' ', '_').replace(':', '_')
    os.makedirs(path, exist_ok=True)
    
    with open(f"{path}/raw_{name}.txt", 'wb') as fp:
        fp.write(req.content)
        print(f"Retrieved \'{row['title']}\'")

folder_name = 'data'

df = pd.read_csv("sources.csv")
df.apply(lambda x: process_row(folder_name, x), axis=1)
print(f"Downloaded {df.shape[0]} books")