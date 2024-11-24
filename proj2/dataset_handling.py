'''
    Contains helper functions for loading, splitting, and modifying the dataset
'''
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import random
import os
import time
from p_logging import logger


def load_dataset(config_name='all', data_dir="data") -> pd.DataFrame:
    '''
    Read in dataset from Parquet file
    '''
    return pq.read_table(f"{data_dir}/{config_name}/dataset.parquet").to_pandas()
    # return pd.read_parquet(f"{data_dir}/{config_name}/dataset.parquet")

def get_config_metadata(data_dir="data"):
    path = f"{data_dir}/run_dates.txt"
    if os.path.exists(path):
        with open(path, 'r') as fp:
            return [int(x) for x in fp.readline().split()]
    else:
        return [-1, -1]
def write_config_metadata(data, data_dir="data"):
    path = f"{data_dir}/run_dates.txt"
    with open(path, 'w') as fp:
        fp.write(' '.join([str(x) for x in data]))
        fp.write("\n")



def book_train_test_split(df, test_size=0.2, margin_of_error=0.001, initial_growth=1, growth=1) -> pd.DataFrame:
    '''
    "Splits" the dataset into train and test groups. 

    For simplicity, this is run before feature extraction, so rather than returning 
    4 different dataframes, this function return the input data frame with a boolean column
    called "is_train", which you can very quickly split by.

    TODO: Add contingency to terminate the while loop in case it goes too long
    Args:
        df - input dataframe to process
        test_size - Desired % of the overall data to be market for testing
        margin_of_error - Margin of error for the test size. The splitting is usually never exact.
    '''
    def get_ratio(sub_df, count_df):
        '''
        Gets the ratio of test data rows to overall data
        '''
        return sub_df['text'].sum() / count_df['text'].sum()

    def get_initial_split():
        '''
        Get the initial split of the dataframe, with one book from each 
        author in the "test" dataset dataframe. 
        '''
        sub_df = pd.DataFrame([], columns=['author_id', 'book_id', 'text'])
        # initial population, get at least one book from each author
        for author in count_df['author_id'].unique():
            # pick one random book
            num_books = count_df[count_df['author_id'] == author]['book_id'].max()
            n = 1
            if num_books >= initial_growth:
                n = initial_growth
            else:
                if initial_growth - 2 >= 0:
                    n = initial_growth - 2
            n = initial_growth if num_books > initial_growth else num_books - 1
            for x in random.sample(range(0, num_books), n):
            # rand_book = random.randint(0, num_books)
                book_row = count_df.loc[(count_df['author_id'] == author) & (count_df['book_id'] == x)]
                sub_df = pd.concat([sub_df, book_row])
        return sub_df
    count_df = df.groupby(['author_id', 'book_id']).count().reset_index()
    sub_df = get_initial_split()
    initial_run = True
    processing = True
    ratio_range = (test_size - margin_of_error, test_size + margin_of_error)
    while processing:
        r = get_ratio(sub_df, count_df)
        logger.debug(f"Ratio: {r:.4f} | CI: ({ratio_range[0]:.4f}, {ratio_range[1]:.4f})")
        if r > ratio_range[0] and r < ratio_range[1]:
            logger.debug("Reached target ratio, exiting")
            processing = False # target reached, exit
        elif r < ratio_range[0]:  
            # too little data, add another random book
            n = growth
            new_row = count_df[~(count_df.index.isin(sub_df.index))].sample(n=growth)
            sub_df = pd.concat([sub_df, new_row])
            initial_run = False
        else:
            # data is too big, either regen or take off random book
            if initial_run:
                # regen if this is the first run
                sub_df = get_initial_split()
            else:
                if growth > 1:
                    growth -= 1
                # take off random book
                book_to_remove = sub_df.sample(n=1)
                if len(book_to_remove) == 1:
                    sub_df = sub_df[~(sub_df.index.isin(book_to_remove.index))]
                initial_run = False
        
    train_elements = sub_df[['author_id', 'book_id']].apply(tuple, axis=1)
    df['is_train'] = df[['author_id', 'book_id']].apply(tuple, axis=1).isin(train_elements)
    return df

