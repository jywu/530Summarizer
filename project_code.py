import itertools
from operator import itemgetter
from Queue import PriorityQueue
from numpy import linalg as LA
import numpy as np
import math, os, operator
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
    return get_tf(get_dir_words(path))

def get_tf(tokens):
  '''Creates a dictionary of tokens mapped to frequency in list'''
  freqs = FreqDist(tokens)
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

NYT_IDF = get_idf(NYT_DOCS)
NYT_LEN = len(get_all_files(NYT_DOCS))

def get_tfidf(tf_dict, idf_dict):
  '''Creates dictionary of words mapped to TF-IDF values'''
  tfidf = dict((word, tf_dict[word]*idf_dict.get(word, math.log(NYT_LEN))) for word in tf_dict)
  return tfidf;

def make_tfidf_dict(sentences):
  '''Creates a dictionary mapping all words to TFIDF values, using NYT articles for IDF'''
  words = tokenize_sentences(sentences)
  tf_dict = get_tf(words) 
  return get_tfidf(tf_dict, NYT_IDF)

def score_sentence_TFIDF(sentence, tfidf_dict):
  '''Computes TFIDF score of a sentence'''
  score = 0.0
  words = word_tokenize(sentence)
  for word in words:
    score += tfidf_dict[word]
  return score / len(words)
    

### (potentially) shared methods ###

def summary_length(summary_list):
  '''Calculates total word length of summary, if summary is a list of sentences'''
  return reduce(lambda acc, sen: acc + len(word_tokenize(sen)), summary_list, 0)

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

def create_feature_space(sentences):
    tokens = [word_tokenize(s) for s in sentences]
    vocabulary = set(reduce(lambda x, y: x + y, tokens))
    return dict([(voc, i) for (i, voc) in enumerate(vocabulary)])

def vectorize_w(feature_space, vocabulary,dct):
    vectors = [0] * len(feature_space)
    for word in vocabulary:
        if (feature_space.has_key(word)):
            vectors[feature_space[word]] = dct.get(word, 0)
    return vectors

def vectorize(feature_space, sentence, dct):
    return vectorize_w(feature_space, list(set(word_tokenize(sentence))), dct)

def is_valid(sent, summary, dct, vector=None):
    if len(summary) == 0: return True
    if vector == None: vector = create_feature_space(summary)
    num_words = len(word_tokenize(sent))
    vector_x = vectorize(vector, sent, dct)
    if(num_words < 9 or num_words > 45): #need to determine threshold
        return False;
    for sent in summary:
        vector_y = vectorize(vector, sent, dct)
        sim = cosine_similarity(vector_x, vector_y)
        if(sim > 0.5): #need to determine threshold
            return False
    return True

### LexRank Summarizer ###

THRESHOLD = 0.001

def build_feature_space(all_sents, tfidf_dict):
    feature_space = []
    for sent in all_sents:
        row = []
        for word in tfidf_dict.keys():
            if word in sent:
                row.append(tfidf_dict[word])
            else:
                row.append(0)
        feature_space.append(row)
    return feature_space

def normalize_matrix(matrix):
    for i in range(len(matrix)):
        row = matrix[i]
        sum_value = sum(row)
        if sum_value != 0:
            matrix[i] = [(sim * 1.0 /sum_value) for sim in row]
    return matrix 

def build_similarity_matrix(feature_space):
    length = len(feature_space)
    matrix = [[None for i in range(length)] for j in range(length)]
    for i in range(length):
        for j in range(length):
            if matrix[i][j] == None:
                row1 = feature_space[i]
                row2 = feature_space[j]
                sim = cosine_similarity(row1, row2)
                matrix[i][j] = sim
                matrix[j][i] = sim
    return matrix

def make_graph(feature_space, all_sents):
    matrix = build_similarity_matrix(feature_space)
    length = len(all_sents)
    graph = [[0 for i in range(length)] for j in range(length)]
    for row in range(len(matrix)):
        for col in range(len(matrix)):
            if matrix[row][col] > THRESHOLD:
                graph[row][col] = 1
    return normalize_matrix(graph)

def page_rank_iteration(graph, all_sents):
    # get principle eigenvector
    values, vectors = LA.eig(graph)
    max_index = np.argmax(values)
    # print 'max_index', max_index
    eig_vector = vectors[max_index]
    # print eig_vector
    score_dict = dict()
    for i in range(len(all_sents)):
        score_dict[all_sents[i]] = eig_vector[i]

    # sort according to value, and get top sentences
    sorted_dict = sorted(score_dict.iteritems(), key=itemgetter(1), reverse=True)
    return [entry[0] for entry in sorted_dict]

def get_top_sentences(sents, num_words, tfidf_dict):
    num_token = 0
    result = []
    while(num_token <= 100):
        sent = sents.pop(0)
        if is_valid(sent, result, tfidf_dict):
            result.append(sent)
            num_token += len(word_tokenize(sent))
    return result

def lex_sum_helper(dir_path):
    all_sents = load_collection_sentences(dir_path)
    tfidf_dict = make_tfidf_dict(all_sents)
    feature_space = build_feature_space(all_sents, tfidf_dict)
    # feature_space = create_feature_space(all_sents)
    print 'feature_space!'
    graph = make_graph(feature_space,all_sents)
    print 'graph!'
    sorted_sents = page_rank_iteration(graph, all_sents)
    print 'sorted sents!'
    filtered_sents = get_top_sentences(sorted_sents, 100, tfidf_dict)
    summary = '\n'.join(filtered_sents)
    return summary

def LexRankSum(input_collection, output_folder):
    dir_list = get_sub_directories(input_collection)
    for directory in dir_list:
        print directory
        # generate input and output paths
        dir_path = input_collection + "/" + directory
        output_file = output_folder + "/" + gen_output_filename(directory)
        # create summary and write to file
        summary = lex_sum_helper(dir_path)
        write_to_file(output_file, summary)
    
#LexRankSum(DEV, '../lexPageRank')


### TF-IDF Summarizer ### ROUGE-2 Recall (DEV) = 0.06932
def TFIDFSum(input_collection, output_folder):
  if not input_collection.endswith('/'): input_collection += '/'
  if not output_folder.endswith('/'): output_folder += '/'
  dir_list = get_sub_directories(input_collection)
  for directory in dir_list:
    sentences = load_collection_sentences(input_collection + directory)
    summary = "\n".join(gen_TFIDF_summary(sentences))
    output = output_folder + gen_output_filename(directory)
    write_to_file(output, summary)

def gen_TFIDF_summary(sentences):
  '''Makes TFIDF summary for a list of sentences'''
  summary = []
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
  while summary_length(summary) <= 100 and not pq.empty():
    score, next_sentence = pq.get()
    if is_valid(next_sentence, summary, tfidf_dict):
      summary.append(next_sentence)
  return summary

