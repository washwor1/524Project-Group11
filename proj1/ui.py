import re
import nltk
import contractions
import numpy as np

def normalize_text_block_compact(block: str):
    '''
    Apply various patterns to normalize a block of text
    Compact version that ignores the options in the preprocess_data.py
    '''
    normalizing_patterns = {
        r"^\s*[\s*]+$": "",  # filter out line breaks with asterisk
        r"[“”]": "\"",       # replace Unicode quotes with standard ones
        r"[‘’]": "'",        # replace Unicode apostrophes with standard ones
        r"[–—]": "-",        # replace long dashes with standard dash
        r"\[((.|\n|\r)*?)\]$": "",  # remove notes at the end
        r"\[((.|\n|\r)*?)\]": "",   # remove inline notes
        r"\|(.*)\|": "",     # remove text between bars
        r"\s+\+-+\+": "",    # remove ASCII art headers
        r"^\s+\.{5,}((.|\n|\r)*?)\.{5,}.*$": "",  # remove patterns of dots
        r"\n{3,}": "\n\n",   # shorten large margins
        r"\"": "",           # remove quotes
        r"--": "",           # remove double dashes
        r"^End of Project Gutenberg's .*$": "",  # remove specific ending line
        r"(\d{4,}|\d{1,3}-\d{1,3}-\d{1,3})": "", # remove years and long numbers
        r"\d{2,}\.\d{2}": "",  # remove times formatted as hh.mm
        r"\d{1,}(th|st|nd|rd)": "",  # remove ordinal numbers
        r"(\$|£)?\s?([0-9]{1,3},)*[0-9]{1,3}": "",  # remove currency amounts
        r"vi{2,}": ""        # remove repeated 'vi'
    }

    # Apply normalization patterns
    for pat, sub in normalizing_patterns.items():
        block = re.sub(pat, sub, block, flags=re.MULTILINE)

    # Strip whitespace and fix contractions
    block = block.strip()
    block = contractions.fix(block)

    # Tokenize and convert to lowercase without removing stopwords or punctuation
    tokens = [token.lower() for token in nltk.word_tokenize(block)]

    # Join tokens back into a string
    res = ' '.join(tokens)
    return res

def average_embeddings(embeddings):
    '''
    Simple function to average glove embeddings for a sequence
    '''
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        #Returns zeros if the embeddings are empty for some reason. 
        return np.zeros(300)  

def get_document_embedding(words, embeddings_index, averaging_function=average_embeddings):
    '''
    Compact document embeddings for user input.
    '''
    embeddings = []
    for word in words:
        embedding = embeddings_index.get(word)
        if embedding is not None:
            embeddings.append(embedding)
        else:
            continue  
    return averaging_function(embeddings)

def run_ui(input_text, embeddings_index, model):
    text = normalize_text_block_compact(input_text)
    embed = get_document_embedding(text, embeddings_index)
    prediction = model.predict(embed.reshape(1, -1))
    print(f'Model Prediction for Input Sequence: {prediction}')


