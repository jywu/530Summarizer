import os, math
from Queue import PriorityQueue
from nltk import FreqDist
from nltk.corpus import PlaintextCorpusReader
from nltk import word_tokenize

### GLOBAL VARIABLES ###
DEV = '/home1/c/cis530/final_project/dev_input/'
TEST = '/home1/c/cis530/final_project/test_input/'
NYT_DOCS = '/home1/c/cis530/final_project/nyt_docs/'

### Loading from Files (from HW1) ###
def get_all_files(path):
  '''Returns a list of all files in path'''
  files_all = PlaintextCorpusReader(path, '.*')    
  return files_all.fileids()
def get_sub_directories(path):
  return [filename for filename in os.listdir(path) if os.path.isdir(os.path.join(path, filename))]
def load_file_sentences(filename):
  '''Returns a list of lowercased sentences in file. Assumes file format is one sentence per line'''
  f = open(filename, 'r')
  sentences = f.readlines()
  f.close()
  return [sen.lower().strip() for sen in sentences]  
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
    tokens.extend(word_tokenize(s))
  return tokens


### TF-IDF (from HW1 MODIFIED) ###

def get_tf_path(path):
  '''Creates a dictionary of words in path mapped to term frequency'''
  get_tf(get_dir_words(path))


def get_tf(tokens):
  '''Creates a dictionary of tokens mapped to frequency in list'''
  freqs = FreqDist(tokens)
  word_count = sum(freqs.values())
  df_dict = dict((x, (freqs[x]+0.0)) for x in freqs)
  return df_dict
           
def get_idf(directory):
  '''Creates dictionary of words mapped to IDF'''
  df_dict = {};
  full_vocab = list(set(load_collection_tokens(directory)))
  for vocab in full_vocab: df_dict[vocab] = 1        
  files = get_all_files(directory); N = len(files)
  for eachfile in files:
    tokens = list(set(load_file_tokens(os.path.join(directory, eachfile))))
    for token in tokens: df_dict[token] += 1;    
  idf_dict = dict((word, math.log(N/df_dict[word])) for word in df_dict)
  return idf_dict;    

def get_tfidf(tf_dict, idf_dict):
  '''Creates dictionary of words mapped to TF-IDF values'''
  tfidf = dict((word, tf_dict[word]*idf_dict.get(word, 1)) for word in tf_dict)
  return tfidf;

def make_tfidf_dict(sentences):
  '''Creates a dictionary mapping all words to TFIDF values, using NYT articles for IDF'''
  words = tokenize_sentences(sentences)
  tf_dict = get_tf(words)
  idf_dict = get_idf(NYT_DOCS)  
  return get_tfidf(tf_dict, idf_dict)

def score_sentence_TFIDF(sentence, tfidf_dict):
  '''Computes TFIDF score of a sentence'''
  score = 0.0
  words = word_tokenize(sentence)
  for word in words:
    score += tdidf_dict(word)
  return score / len(words)
    

### (potentially) shared methods ###

def gen_output_filename(directory):
    '''Creates an output file name by adding sum_ to input dir name'''
    return 'sum_' + directory + '.txt'

def write_to_file(file_path, summary):
    '''Writes the given summary to file'''
    f = open(file_path, 'w')
    f.write(summary)
    f.close()

def add(x, y): return x + y

def cosine_similarity(vectorX, vectorY):
    numerator = 0
    for i in range(len(vectorX)):
        numerator += vectorX[i] * vectorY[i]
    denom_v = [v * v for v in vectorX]
    denom_w = [w * w for w in vectorY]
    denom = math.sqrt(reduce(add, denom_v) * reduce(add, denom_w))
    if denom != 0:
        return numerator / float(denom)
    else:
        return 0
    return result

def is_valid(sent, summary, vector, dct):
    num_words = len(word_tokenize(sent))
    vector_x = vectorize(vector, sent, dct)
    if(num_words < 9 or num_words > 45): #need to determin threshold
        return False;
    for sent in summary:
        vector_y = vectorize(vector, sent, dct)
        sim = cosine_similarity(vector_x, vector_y)
        if(sim > 0.5): #need to determin threshold
            return False
    return True

### LexRank Summarizer ###

def LexRankSum(input_collection, output_folder):
    dir_list = get_sub_directories(input_collection)
    for directory in dir_list:
        # generate input and output paths
        dir_path = input_collection + "/" + directory
        output_file = output_folder + "/" + gen_output_filename(directory)
        # create summary and write to file
        # summary = lex_sum_helper(dir_path)
        # write_to_file(output_file, summary)
    
# LexRankSum('/home1/c/cis530/final_project/dev_input/', '..')




### TF-IDF Summarizer ###
def TFIDFSum(input_collection, output_folder):
  if not input_collection.endswith('/'):
    input_collection += '/'
  if not output_folder.endswith('/'):
    output_folder += '/'
  dir_list = get_sub_directories(input_collection)
  for directory in dir_list:
    sentences = load_collection_sentences(input_collection + directory)
    summary = gen_TFIDF_summary(sentences)
    output = output_folder + gen_output_filename(directory)
    write_to_file(output, summary)

def gen_TFIDF_summary(sentences):
  '''Makes TFIDF summary for a list of sentences'''
  summary = ""
  tfidf_dict = make_tfidf_dict(sentences)

  #Calculate sentence scores; negate so that higher TFIDFs show up first in PQ
  neg_scores = []
  for sentence in sentences: 
    neg_scores.append(-score_sentence_TFIDF(sentence, tfidf_dict))

  #Rank sentences
  pq = PriorityQueue()  
  for pair in zip(neg_scores, sentences):
    pq.put(pair)

  #Make greedy summary
  while len(summary) <= 100 and not pq.empty():
    score, next_sentence = pq.get()
    if is_valid(next_sentence):
      summary += next_sentence + '\n'
  return summary

