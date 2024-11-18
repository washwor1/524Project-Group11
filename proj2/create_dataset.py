'''
Create Dataset Script

Contains functions to download raw data, process it, and combine it into a single parquet file
'''


import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zipfile
import logging 
import gdown

from tqdm.auto import tqdm

# Register `pandas.progress_apply` and `pandas.Series.map_apply` with `tqdm`
# (can use `tqdm.gui.tqdm`, `tqdm.notebook.tqdm`, optional kwargs, etc.)
tqdm.pandas(desc="my bar!")

logger = logging.getLogger("dataset_logger")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)

def unzip_archive(file, output_path):
    with zipfile.ZipFile(file, "r") as zip_fp:
        zip_fp.extractall(output_path)

def download_gdrive_file(file_id, output_file):
    '''
    Downloads dataset stored on Google Drive
    '''
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_file, quiet=False)

def process_metadata(metadata_path="data/spgc/metadata/metadata.csv"):
    '''
    Apply filters to the metadata to downselect to the best number of books
    '''
    logger.info("Processing metadata...")
    df = pd.read_csv(metadata_path)
    # include English-only books (Leaves ~59633 and removes ~14910)
    df['is_english'] = df['language'].apply(lambda x: x == "['en']")
    # remove non-text entries and non-English books (leaves 58,477)
    df = df[df['is_english'] & (df['type'] == 'Text')]
    df['pg_code'] = df['id'].apply(lambda x: int(x[2:]))
    # remove books with author as "Various" or "Anonymous" or "Unknown" (removes 4353 + 118)
    df = df[~(df['author'] == 'Various') & ~(df['author'] == 'Anonymous') & ~(df['author'] == 'Unknown')]
    # as of now, I am going to keep books with more than one author and just treat the group as one author. We can fix this later 
    # if this confounds the model
    df = df.dropna(axis=0, subset=['author', 'title'])
    df = df.sort_values('pg_code').reset_index(drop=True)
    df['author'] = df['author'].astype('category')
    df['author_id'] = df['author'].cat.codes
    df['book_id'] = df.groupby('author_id').cumcount()
    # remove authors with only one book (19339 authors -> 5899 authors; 51720 books -> 38280 books)
    book_counts = df.groupby('author_id')['book_id'].count()
    df = df[df['author_id'].isin(book_counts[book_counts > 1].index)]
    meta_df = df.drop(columns=['id', 'authoryearofbirth', 'authoryearofdeath', 'language', 'downloads', 'subjects', 'type', 'is_english']).reset_index(drop=True)
    logger.info("Finished processing metadata...")
    return meta_df

def process_dataset(meta_df, data_dir="data/spgc/data/tokens"):
    '''
    Process all the selected works in SPGC into a single Parquet file
    
    This will take a while. 
    '''
    arr = [] # columns: author_id, book_id, tokens
    failed_arr = [] # array of book codes that could not be found
    CHUNK_SIZE = 500
    def read_book(row):
        # read in file
        nonlocal arr
        nonlocal failed_arr
        tokens = []
        try:
            with open(f"{data_dir}/PG{row['pg_code']}_tokens.txt", 'r', encoding='utf-8') as fp:
                tokens = fp.read().splitlines()
        except FileNotFoundError:
            failed_arr.append(row['pg_code'])
        
        # split into N-sized chunks
        token_chunks = [[row['author_id'], row['book_id'], ' '.join(tokens[i:(i + CHUNK_SIZE)])] for i in range(0, len(tokens), CHUNK_SIZE)]
        arr += token_chunks
        # logger.info(f"Processed book {row.name}")
    meta_df.progress_apply(read_book, axis=1)
    logger.info("Finished Processing Dataset")

    data_df = pd.DataFrame(arr, columns=['author_id', 'book_id', 'text'])
    tbl = pa.Table.from_pandas(data_df)
    pq.write_table(tbl, 'data/dataset.parquet')

    # remove missing records from metadata so only included books are recorded
    meta_df = meta_df[~meta_df['pg_code'].isin(failed_arr)].reset_index(drop=True)
    meta_df.to_csv("data/metadata.csv", index=False)

if __name__ == "__main__":
    # code to download dataset
    # will download an ~8 GB file, so it will take second
    # download_gdrive_file("1VJcL_0B-7YcAkaSTXnHOKXLa_EAbmpCK", "data/spgc_raw.zip")
    # unzip_archive("data/spgc_raw.zip", "data/spgc/")
    
    # apply misc. filters to metadata to select certain works
    meta_df = process_metadata()
    
    # combine all three files into a single parquet file
    process_dataset(meta_df)