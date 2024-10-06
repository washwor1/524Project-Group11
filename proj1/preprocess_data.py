'''
    Preprocess Data Script

    Usage: 
        This script takes the data and performs some initial preprocessing on it to 
        convert it from heavily formatted UTF-8 text files to parsable chunks.
    Remarks:
        I've tried my best to make a generalized solution that'll work for all 16 chosen
        works. There are a few cases where the structure differs widely across books, in 
        particular with "The Blonde Lady" and "A Study in Scarlet" which both have "sections"
        that contain chapters, whereas all the other books have "chapters"
    
    TODO:
        - "The Adventures of Sherlock Holmes" - Detect subsections (each "chapter" is a novella)
            with sections of its own so splitting that is important to be accurate

'''
import re
import os
import glob
import pandas as pd
import sys

#region RegEx Patterns

# filter front/rear matter
matter = re.compile(r"\*{3} (START|END) OF THE PROJECT GUTENBERG .+ \*{3}$", re.MULTILINE | re.IGNORECASE)
# extract table of contents 
contents = re.compile(r"(Contents|CONTENTS)([\s\S]*?)(\n{4})", re.MULTILINE)
# extract section from ToC (only in "A Study in Scarlet" and "A Blonde Lady")
toc_section = re.compile(r"^\s*(PART (I[XV]|V?I{0,3})|(FIRST|SECOND) EPISODE)[.:](.*)$", re.MULTILINE)
# extract chapter entry from ToC (can have roman numerals or numbers)
toc_chapter = re.compile(r"^\s*(CHAPTER|Chapter)?\s*((?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})|\d{1,3})\.? (.*?)\d{0,3}$", re.MULTILINE)

# book-specific patterns to detect chapters (most have their own formatting scheme)
# - For "The Blonde Lady" and "A Study in Scarlet", the patterns also match parts/episodes
patterns = {
    "data\\agatha_christie\\raw_poirot_investigates.txt": r"  (?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})\n{1,3}\s+(.*)",
    "data\\agatha_christie\\raw_the_murder_of_roger_ackroyd.txt": r"CHAPTER (?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})\n\n\s+{00}",
    "data\\agatha_christie\\raw_the_murder_on_the_links.txt": r"^\d{1,2} ",
    "data\\agatha_christie\\raw_the_mysterious_affair_at_styles.txt": r"^CHAPTER \w+\.\n",
    "data\\gk_chesterton\\raw_the_innocence_of_father_brown.txt": r"{00}",
    "data\\gk_chesterton\\raw_the_man_who_knew_too_much.txt": r"^(?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})\. ",
    "data\\gk_chesterton\\raw_the_man_who_was_thursday__a_nightmare.txt": r"^CHAPTER \w+\.\n{00}",
    "data\\gk_chesterton\\raw_the_wisdom_of_father_brown.txt": r"^\w{3,6}\.? -- {00}",
    "data\\maurice_leblanc\\raw_the_blonde_lady.txt": r"((FIRST|SECOND) EPISODE|(CHAPTER (\w+)))\n\n{00}", # CHAPTER (\w+)\n\n{00}
    "data\\maurice_leblanc\\raw_arsene_lupin_super-sleuth.txt": r"^CHAPTER \w+\n\n(.*)\n",
    "data\\maurice_leblanc\\raw_arsene_lupin.txt": r"^CHAPTER \w+\n(.*)\n",
    "data\\maurice_leblanc\\raw_the_extraordinary_adventures_of_arsene_lupin_gentleman-burglar.txt": r"^(?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})\. (.*)",
    "data\\sir_arthur_conan_doyle\\raw_a_study_in_scarlet.txt": r"(PART|CHAPTER) \w{1,3}.\n_?{00}\.?_?",
    "data\\sir_arthur_conan_doyle\\raw_the_adventures_of_sherlock_holmes.txt": r"^(?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})\. {00}",
    "data\\sir_arthur_conan_doyle\\raw_the_hound_of_the_baskervilles.txt": r"^Chapter \d+\.\n{00}\n",
    "data\\sir_arthur_conan_doyle\\raw_the_sign_of_the_four.txt": r"^Chapter \w+\n{00}\n"
}
#endregion

def get_files(data_dir='data') -> [str]:
    '''
    Gets the raw file paths to be used

    Arguments:
        data_dir - Path to data/ directory (default to cwd)
    Returns:
        Array of file names
    '''
    return glob.glob(f'{data_dir}/**/raw*.txt') 



def extract_structure(toc: str) -> pd.DataFrame:
    '''
    Processes Table of Contents (ToC) to extract the structure
    of the underlying text
    
    Returns:
        DataFrame with section number, chapter number, and name columns 
        (rows with section number of 0 and chapter number of 0 denote sections)
    '''
    # extract "sections" first from ToC (defaults to whole string)
    sections = {
        0: {
            "start": 0,
            "end": len(toc),
            "match_obj": None
        }
    }
    if matches := list(re.finditer(toc_section, toc)):
        sections[0] = {
            "start": matches[0].end(),
            "end": matches[1].start(),
            "match_obj": matches[0]
        }
        sections[1] = {
            "start": matches[1].end(),
            "end": len(toc),
            "match_obj": matches[1]
        }
    
    arr = []
    for idx, sec in sections.items():
        test = toc[sec["start"]:sec["end"]]
        # append Chapter 0 (i.e. the section itself)
        if t := sec['match_obj']:
            arr.append([idx, 0, t.group(4).strip()])
        # extract chapters
        ch_arr = [
            [idx, i + 1, ch.group(6).strip()]
            for i, ch in enumerate(re.finditer(toc_chapter, test) or [])
        ]
        # fallback - treat each nonempty line as an entry
        #   - used in "The Innocence of Father Brown"
        if len(ch_arr) == 0:
            i = 1
            for x in toc.split('\n'):
                if y := x.strip():
                    ch_arr.append([idx, i, y])
                    i += 1
        arr += ch_arr
    return pd.DataFrame(arr, columns=['section_num','chapter_num', 'name'])

def find_match(pattern: str, name: str, text: str) -> (int, int):
    '''
    Find a RegEx match in text

    Returns:
        Start and end index of match
    Raises:
        Exception if no match can be found
    '''
    r = re.compile(pattern.replace("{00}", name), re.MULTILINE | re.IGNORECASE)
    if s := re.search(r, text):
        return s.start(), s.end()
    
    raise Exception(f"Could not parse pattern \'{r.pattern}\'")    


def split_txt(txt, df, ch_pattern, file_name='') -> pd.DataFrame:
    '''
    Split text into sections and chapters.
    '''
    output_dir = file_name.replace("raw_", "").replace(".txt", "/").replace(',', '')
    os.makedirs(output_dir, exist_ok=True)

    # process section (there should be at least one per text)
    # this array is reversed just in case there is >1 section, 
    # so we can subset the text correctly
    sections = df['section_num'].unique()[::-1]
    i = len(sections)
    arr = []
    for s in sections:
        # identify 0th row (if necessary) and subset text to include it
        subset = txt
        section_arr = []
        header = df[(df['section_num'] == s) & (df['chapter_num'] == 0)]
        sub_df = df[(df['section_num'] == s) & (df['chapter_num'] != 0)].reset_index(drop=True)
        
        if len(header) != 0:
            title = header.iloc[0]['name']
            a, b = find_match(ch_pattern, title, subset)
            subset = txt[b:] # subset section for processing
            txt = txt[:a]
        
        # split section into chapters
        for idx, row in sub_df.iterrows():
            name = row['name'].strip().replace('  ', ' ')
            
            formatted_name = name.upper().replace('[', '\[').replace(']', '\]')
            a, b = find_match(ch_pattern, formatted_name, subset)

            if idx > 0: 
                section_arr.append(subset[:a]) 
            subset = subset[b:]
        section_arr.append(subset) # append remaining subset to array
       
        # write processed text to data directory for debugging and visualizing results
        with open(f"{output_dir}/section_{i}.txt", "w", encoding='utf-8') as fp:
            pad = '-' * 25
            if len(header) != 0:
                fp.write(f"{pad} SECTION {header.iloc[0]['name']} {pad}\n")
            for idx, x in enumerate(section_arr):
                ch_name = sub_df.iloc[idx]['name']
                fp.write(f"{pad} BEGIN CHAPTER {ch_name} {pad}\n")
                fp.write(x)
                fp.write(f"\n{pad} END CHAPTER {ch_name} {pad}\n")
            i -= 1
        
        if len(header) != 0: 
            section_arr.insert(0, '')
        arr = section_arr + arr

    df['text'] = arr
    # very naive word count (without much preprocessing)
    df['wc'] = df['text'].apply(lambda x: len([y.strip() for y in x.split(" ") if y.strip()]))
    df.to_csv(f"{output_dir}/data.csv", index=False)
    return df



def process_file(file_name: str):
    raw = ''
    with open(file_name, 'r', encoding='utf-8') as fp:
        raw = fp.read()
    print(file_name)
    # remove front and end matter
    x = list(re.finditer(matter, raw))
    txt = raw[x[0].end():x[1].start()]
    # extract stuctural information
    blocks = [txt]
    if search := re.search(contents, txt):
        toc = search.group(2)
        df =  extract_structure(toc)
        test = txt[search.end():]
        split_txt(test, df, patterns.get(file_name.replace('/', '\\')), file_name)

if __name__ == "__main__":
    try:
        args = sys.argv
        if len(args) > 1:
            process_file(args[1])
        else:
            files = get_files()
            for f in files:
                process_file(f)
    except Exception:
        print("ERROR")
        raise