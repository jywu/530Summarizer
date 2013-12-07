import numpy
import os
import math
from nltk import FreqDist
from nltk.corpus import PlaintextCorpusReader
from nltk import sent_tokenize, word_tokenize

ROOT_DIR = '/home1/c/cis530/final_project/'
DEV_INPUT = '/home1/c/cis530/final_project/dev_input/'
DEV_MODELS = '/home1/c/cis530/final_project/dev_models/'
TEST_INPUT = '/home1/c/cis530/final_project/test_input/'
NYT_DOCS = '/home1/c/cis530/final_project/nyt_docs/'

### Loading from Files (from HW1) ###
def get_sub_directories(directory):
    sub_dirs = os.listdir(directory)
    if '.DS_Store' in sub_dirs:
        sub_dirs.remove('.DS_Store')
    return sub_dirs

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



### potentially shared methods ###
def gen_output_filename(directory):
    '''Creates an output file name by adding sum_ to input dir name'''
    return 'sum_' + directory + '.txt'

def write_to_file(file_path, summary):
    '''Writes the given summary to file'''
    f = open(file_path, 'w')
    f.write(summary)
    f.close()

def add(x, y): return x+y

def cosine_similarity(vectorX, vectorY):
    '''Calculates cosine similarity between two vectors'''
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
    '''Checks if a sentence is valid'''
    num_words = len(word_tokenize(sent))
    vector_x = vectorize(vector, sent, dct)
    if(num_words < 9 or num_words > 45):
        return False;
    for sent in summary:
        vector_y = vectorize(vector, sent, dct)
        sim = cosine_similarity(vector_x, vector_y)
        if(sim > 0.5):
            return False
    return True

### LexRank Summarizer ###

THRESHOLD = 0.5
STOP_LEVEL = 0.1

def get_file_sent_pairs(dir_path):
    '''Builds a list of (file, sentence) tuples'''
    files = get_all_files(dir_path)
    for f in files:
        file_path = dir_path + '/' + f
        file_sents = load_file_sentences(file_path)
        for sent in file_sents:
            all_sents += (f, sent)
    return all_sents

def get_file_token_tf_dict(dir_path):
    '''Builds a tf dictionary of (f, word):tf '''
    files = get_all_files(dir_path)
    tf_dict = dict()
    for f in files:
        file_path = dir_path + '/' + f
        file_tf_dict = get_tf_path(file_path)
        for key in file_tf_dict.keys():
            newkey = (f, key)
            value = file_tf_dict[key]
            tf_dict[newkey] = value
    return tf_dict

def build_feature_space(all_sents, all_words):
    feature_space = []
    for sent in all_sents:
        row = []
        for word in all_words:
            if word[1] in sent[1]:
                row += tfidf_dict[word]
            else:
                row += 0
        feature_space += row
    return feature_space

def make_graph(feature_space):
    pairs = list(itertools.combinations(feature_space, 2))
    graph = dict((pair, []) for pair in pairs)
    for pair in pairs:
        similarity = consine_similarity(pair[0], pair[1])
        if similarity > THRESHOLD:
            graph[pair[0]] += pair[1]
            graph[pair[1]] += pair[0]
    return graph

def page_rank_iteration(graph, all_sents):
    score_dict = dict((sent[1], (1, 0)) for sent in sents)
    diff = 1 
    while(diff < STOP_LEVEL):
        # calculate share
        for tupl in all_sents:
            sent = tupl[1]
            neighbours = graph[sent]
            share = score_dict[sent][0] * 1.0 / len(neighbours)
            share_dict[sent][1] = share
            for neighbour in neighbours:
                score_dict[neighbour][1] += share
        # update score, calc diff, reset share
        diff = 0
        for key in score_dict.keys():
            share = score_dict[key][1]
            score_dict[key][0] += share
            diff += share * share
            score_dict[key][1] = 0
        diff = math.sqrt(diff)
    # sort according to value, and get top sentences
    sorted_dict = sorted(score_dict.iteritems(), key=itemgetter(1)[0], reverse=True)
    return [pair[0] for pair in sorted_dict]

def get_top_sentences(sents, num_words):
    num_token = 0
    result = []
    for sent in sents:
        num_token += len(word_tokenize(sent))
        result += sent
        if num_token > 0:
            result -= sent
            break
    return result

def lex_sum_helper(dir_path):
    tf_dict = get_file_token_tf_dict(dir_path)
    tfidf = dict((word, tf_dict[word]*idf_dict.get(word[1], 1)) for word in tf_dict)
    all_sents = get_file_sent_pairs(dir_path)
    all_words = tfidf_dict.keys()
    feature_space = build_feature_space(all_sents, all_words)
    graph = make_graph(feature_space) 
    filtered_sents = page_rank_iteration(graph, all_sents)
    summary = '\n'.join(filtered_sents)
    return summary
    
def LexRankSum(input_collection, output_folder):
    dir_list = get_sub_directories(input_collection)
    for directory in dir_list:
        # generate input and output paths
        dir_path = input_collection + "/" + directory
        output_file = output_folder + "/" + gen_output_filename(directory)
        # create summary and write to file
        summary = lex_sum_helper(dir_path)
        write_to_file(output_file, summary)
    
LexRankSum('/home1/c/cis530/final_project/dev_input/', '..')

