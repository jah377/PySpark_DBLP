import re
import string
import unicodedata


def json2dict(path):
    '''
    used to import jsons into python
    '''
    x = pd.read_json(path)
    return dict(zip(x['id'], x['name']))


def remove_accents(s):
    '''
    https://docs.python.org/3/library/unicodedata.html#module-unicodedata
    '''
    s = str(s)      # cannot assume string
    norm_form = unicodedata.normalize('NFKD', s)
    return "".join([char for char in norm_form if not unicodedata.combining(char)])


def process_str(s, key=False):
    '''
    processes strings for fn: compare_str_overlap 
    '''
    s = str(s)
    s = s.lower()
    s = s.replace('\\', r'\\')
    s = remove_accents(s)
    s = s.strip()  # remove leading/trailing white space

    if key == True:
        return s.split(r'/')[-1]

    s = re.sub(r'['+string.punctuation+']', ' ', s)
    s = s.split()  # split by white space

    return s


def compare_str_overlap(pauthor, ptitle, pkey):
    '''
    intuition:
        - pkey contains author name
        - pauthor or ptitle values may be switched
        - strings processed and tokenized 
        - column containing string with most overlap to key identifies location of author value 
        - if ptitle has more overlap -- must switch valuess with pauthor

    Output:
        - bool:     True if ptitle has more overlap - must be switched
    '''

    pauthor = process_str(pauthor, key=False)
    ptitle = process_str(ptitle, key=False)
    pkey = process_str(pkey, key=True)

    max_titleOverlap = max([len(re.findall(substr, pkey)[0])
                            for substr in ptitle
                            if re.findall(substr, pkey)])

    max_authorOverlap = max([len(re.findall(substr, pkey)[0])
                             for substr in pauthor
                             if re.findall(substr, pkey)])

    return max_titleOverlap > max_authorOverlap
