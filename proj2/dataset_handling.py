'''
    Contains helper functions for loading, splitting, and modifying the dataset
'''
import logging
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import random

def load_dataset(data_path='data/dataset.parquet') -> pd.DataFrame:
    '''
    Read in dataset from Parquet file
    '''
    tbl = pq.ParquetFile(data_path)
    return tbl.read().to_pandas()


def book_train_test_split(df, test_size=0.2, margin_of_error=0.001) -> pd.DataFrame:
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
            rand_book = random.randint(0, num_books)
            book_row = count_df.loc[(count_df['author_id'] == author) & (count_df['book_id'] == rand_book)]
            sub_df = pd.concat([sub_df, book_row])
        return sub_df
    
    count_df = df.groupby(['author_id', 'book_id']).count().reset_index()
    sub_df = get_initial_split()
    initial_run = True
    processing = True
    ratio_range = (test_size - margin_of_error, test_size + margin_of_error)
    while processing:
        r = get_ratio(sub_df, count_df)
        if r > ratio_range[0] and r < ratio_range[1]:
            processing = False # target reached, exit
        elif r < ratio_range[0]:  
            # too little data, add another random book
            new_row = count_df[~(count_df.index.isin(sub_df.index))].sample(n=1)
            sub_df = pd.concat([sub_df, new_row])
            initial_run = False
        else:
            # data is too big, either regen or take off random book
            if initial_run:
                # regen if this is the first run
                sub_df = get_initial_split()
            else:
                # take off random book
                sub_df = sub_df[~(sub_df.index == sub_df.sample(n=1).index)]
                initial_run = False
    train_elements = sub_df[['author_id', 'book_id']].apply(tuple, axis=1)
    df['is_train'] = df[['author_id', 'book_id']].apply(tuple, axis=1).isin(train_elements)
    return df

