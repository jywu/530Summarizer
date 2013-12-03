import os
from nltk import FreqDist
from nltk.corpus import PlaintextCorpusReader
from nltk import sent_tokenize, word_tokenize

### Loading from Files (from HW1) ###
def get_all_files(path):
  '''Returns a list of all files in path'''
  files_all = PlaintextCorpusReader(path, '.*')    
  return files_all.fileids()
def load_file_sentences(filename):
  '''Returns a list of lowercased sentences in file'''
  fullstring = open(filename, 'r').read()        
  return [sen.lower() for sen in sent_tokenize(fullstring)]  
def load_collection_sentences(path):
  '''Returns a list of lowercased sentences in path'''
  lists = []
  for f in get_all_files(path):
    lists.extend(load_file_sentences(os.path.join(path, f)))
  return lists
def load_file_tokens(filename):
  '''Returns list of tokens in file'''
  linestring = open(filename, 'r').read()
  return [token.lower() for token in word_tokenize(linestring)]    
def load_collection_tokens(path):
  '''Returns list of tokens in path'''
  lists = []
  for f in get_all_files(path):
    lists.extend(load_file_tokens(os.path.join(path,f)))
  return lists
def get_dir_words(path):
  if (os.path.isdir(path)):
    return load_collection_tokens(path);
  return load_file_tokens(path)
def tokenize_sentences(sentence_list):
  '''Returns a list of all tokens'''
  tokens = []
  for s in sentence_list:
    toks.extend(word_tokenize(s))
  return tokens


### TF-IDF (from HW1) ###

def get_tf(path):
  '''Creates a dictionary of words mapped to term frequency'''
  freqs = FreqDist(get_dir_words(path))
  word_count = sum(freqs.values())
  df_dict = dict((x, (freqs[x]+0.0)) for x in freqs)
  return df_dict
           
def get_idf(directory):
  '''Creates dictionary of words mapped to IDF'''
  df_dict = {};
  full_vocab = list(set(load_collection_tokens(directory)))
  for vocab in full_vocab: df_dict[vocab] = 0        
  files = get_all_files(directory); N = len(files)
  for eachfile in files:
    tokens = list(set(load_file_tokens(os.path.join(directory, eachfile))))
    for token in tokens: df_dict[token] += 1;    
  idf_dict = dict((word, math.log(N/df_dict[word])) for word in df_dict)
  return idf_dict;    

def get_tfidf(tf_dict, idf_dict):
  '''Creates dictionary of words mapped to TF-IDF values'''
  tfidf = dict((word, tf_dict[word]*idf_dict[word]) for word in tf_dict)
  return tfidf;

