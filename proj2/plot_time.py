import pyarrow.parquet as pq
import pyarrow.csv as pc

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

file = "/media/volume/team11data/transformer_preds_and_labels.parquet"

csv_file = 'output.csv'

# Read the Parquet file into a PyArrow Table
table = pq.read_table(file)

# Write the PyArrow Table to a CSV file
pc.write_csv(table, csv_file)

print(f"Parquet file '{file}' has been converted to CSV file '{csv_file}'.")