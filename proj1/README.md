# COSC524 - Team 11 Project 1
Assignment: Authorship Analysis -- Maurice Leblanc
Group Members: Tyler Duckworth, Jay Ashworth, Azizul Zahid, Weilin Ouyang
## Setting Up The Project Environment
With Python 3.11+ installed, run:
```
> python -m venv venv\ 
> .\venv\Scripts\activate.ps1
> pip install -r requirements.txt
> python -m nltk.downloader punkt_tab
```
## Downloading the Data
To download the raw dataset, run `python get_novels.py` in the virtual environment. This will populate a directory `data/` which will store the books by author.

## Scripts
The following scripts are available:
- `run_workflow.py` - Runs a single workflow (defaulted to the optimal settings)
- `run_experiments.py` - Runs a series of experiments meant to find the ideal settings configuration for the experiment. See the file for more details.
- `run_independent.py` - Runs a workflow on the test dataset (must be downloaded to a folder called `ind_test/`)
- `plotting.ipynb` - Jupyter Notebook to create the plots used in the presentation and paper (assumes a binary classification dataset is already created)

Additionally, the pipeline itself is split into individual scripts that you can run individually:
- `preprocess_data.py` - Runs the pre-processing pipeline on the raw dataset, converting it into a tokenized format grouped by paragraph. 
- `feature_engineering.py` - Runs feature extraction to create Word Embedding, TF-IDF, and Word Embeddings weighted by TF-IDF score datasets. (Assumes pre-processed data exists in the `data/` folder)
    - Running this the first time will download a 2GB GloVe word embedding dataset, which can take a while given 
- `modeling.py` - Runs the different model combinations on the different generated datasets with the best settings combination (assumes the previous two scripts have been run)
