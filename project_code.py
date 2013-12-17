# Emily Boggs: emboggs@seas.upenn.edu
# Jingyi Wu: wujingyi@seas.upenn.edu

import itertools
from operator import itemgetter
from Queue import PriorityQueue
import numpy as np
from numpy import linalg as LA
import math, os, operator, subprocess, re
from nltk import FreqDist
from nltk.corpus import PlaintextCorpusReader
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn


### GLOBAL VARIABLES ###
DEV = '/home1/c/cis530/final_project/dev_input/'
DEV_MODELS = '/home1/c/cis530/final_project/dev_models/'
TEST = '/home1/c/cis530/final_project/test_input/'
NYT_DOCS = '/home1/c/cis530/final_project/nyt_docs/'
POSITION_DICT = dict()
TS_FILES = []
STOP = set(stopwords.words('english'))
REDUNDANCY_THRESHOLD = 0.8
CURRENT_DIR = os.getcwd() + "/"
TW_DIR = "topicwords/"

def set_redundancy(t):
  global REDUNDANCY_THRESHOLD
  REDUNDANCY_THRESHOLD = t

### Loading from Files (from HW1) ###
def get_all_files(path):
  '''Returns a list of all files in path'''
  files_all = PlaintextCorpusReader(path, '.*')    
  return files_all.fileids()

def get_sub_directories(path):
  return [filename for filename in os.listdir(path) if os.path.isdir(os.path.join(path, filename))]

def load_file_sentences(filename):
  '''Returns a list of sentences in file. Assumes file format is one sentence per line'''
  f = open(filename, 'r')
  sentences = f.readlines()
  f.close()
  return [sen.strip().lower() for sen in sentences]  

def load_collection_sentences(path):
  '''Returns a list of sentences in path'''
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

def write_idf_to_file(idf_dict, idf_file):
  '''Writes IDF data to file in format readable by read_idf_file'''
  f = open(idf_file, 'w')
  for (word, val) in idf_dict.items():
    f.write(word + "\t" + str(val) + "\n")
  f.close()

def read_idf_file(idf_file):
  '''Reads saved IDF information from file'''
  f = open(idf_file, 'r')
  idf_dict = {}
  for line in f.readlines():
    pair = line.strip("\n").split("\t")
    idf_dict[pair[0]] = float(pair[1])
  f.close()
  return idf_dict
    
if os.path.isfile("nyt_idf.data"):
  NYT_IDF = read_idf_file("nyt_idf.data")
else:
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
  words = set(word_tokenize(sentence))
  for word in words:
    score += tfidf_dict[word]
  return score / len(words)

### other shared methods ###

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
    num_words = len(word_tokenize(sent))
    if num_words < 9 or num_words > 45: #need to determine threshold
        return False;
    if len(summary) == 0: return True
    if vector == None: vector = create_feature_space(summary)
    vector_x = vectorize(vector, sent, dct)
    for sum_sent in summary:
        vector_y = vectorize(vector, sum_sent, dct)
        sim = cosine_similarity(vector_x, vector_y)
        if sim > REDUNDANCY_THRESHOLD: #need to determine threshold
            return False
    return True

### LexRank Summarizer ### ROUGE-2 Recall (DEV) = 0.06668

THRESHOLD = 0.1

def build_feature_space(all_sents, tfidf_dict):
    feature_space = []
    for sent in all_sents:
        row = []
        for word in tfidf_dict.keys():
            if word in sent: row.append(tfidf_dict[word])
            else: row.append(0)
        feature_space.append(row)
    return feature_space

def normalize_matrix(matrix):
    #for i in range(len(matrix)):
    #    row = matrix[i]
    #    sum_value = sum(row)
    #    if sum_value != 0:
    #        matrix[i] = [(sim * 1.0 / sum_value) for sim in row]
    for i in range(len(matrix)):
        column = [row[i] for row in matrix]
        sum_value = sum(column)
        if sum_value != 0:
            for row in matrix:
                row[i] = row[i] * 1.0 /sum_value
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
    # return normalize_matrix(graph)
    return graph

def column_value_positive(index, vectors):
    for row in vectors:
        if row[index] <= 0:
            return False
    return True

def get_eigenvector(graph):
    # print graph
    values, vectors = LA.eig(normalize_matrix(graph))
    max_index = np.argmax(values)
    # print values
    # print vectors 
    # print 'max_index', max_index
    if column_value_positive(max_index, vectors):
        # print 'Using vector!!!!!!!'
        eig_vector = []
        for i in range(len(vectors)):
            eig_vector.append(vectors[i][max_index])
    else:
        # print 'Using sum of row!!!!!'
        eig_vector = [sum(row) for row in graph]
    # print 'eig_vector'
    # print eig_vector
    return eig_vector

def page_rank_iteration(graph, all_sents):
    # sums = [sum(row) for row in graph]
    # score_dict = dict(zip(all_sents, sums))
    eig_vector = get_eigenvector(graph)
    # sort according to value, and get top sentences
    score_dict = dict()
    for i in range(len(all_sents)):
        score_dict[all_sents[i]] = eig_vector[i]
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
  summarize(input_collection, output_folder, 2)
    
# LexRankSum(DEV, '../lexPageRank')

### TF-IDF Summarizer ### ROUGE-2 Recall (DEV) = 0.07807
def TFIDFSum(input_collection, output_folder):
  set_redundancy(0.8)
  summarize(input_collection, output_folder, 1)

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
  return "\n".join(summary)


### Greedy KL Summarizer ### current ROUGE-2 = 0.08899
def KLSum(input_collection, output_folder):
  #set_redundancy(1.0)
  summarize(input_collection, output_folder, 3)

def gen_KL_summary(sentences):
  summary = []
  summary_words = []
  summary_freqs = {}

  tokenized = [filter(lambda w: w not in STOP, word_tokenize(s)) for s in sentences]
  sent_freqs = [make_unigram_dict(t) for t in tokenized]

  ## Make distribution Q
  all_tokens = [token for sentence in tokenized for token in sentence] #flattens list
  input_freqs = make_unigram_dict(all_tokens)
  input_probs = dict([(word, input_freqs[word] / len(all_tokens)) for word in input_freqs.keys()])
  input_tfidf = make_tfidf_dict(sentences)

  while summary_length(summary) <= 100 and len(tokenized) > 0:
    ## find sentence with minimum KL
    kl_vals = []
    for i in range(0, len(tokenized)):
      length = len(summary_words) + len(tokenized[i]) 
      kl_vals.append(calculate_KL(summary_freqs, sent_freqs[i], length, input_freqs))
    min_index = kl_vals.index(min(kl_vals))
    
    ## Remove from list and add to summary if valid
    to_add = sentences.pop(min_index)
    to_add_words = tokenized.pop(min_index)
    to_add_freqs = sent_freqs.pop(min_index)
    if is_valid(to_add, summary, input_tfidf):  
      summary.append(to_add)
      summary_words.extend(to_add_words)
      update(summary_freqs, to_add_freqs)
  return "\n".join(summary)

def calculate_KL(p_sum, p_sent, length, q):
  '''Calculates KL divergence between P (summary) and Q (input)'''
  '''Caller provides two frequency dicts for P: one for the summary, and one for the sentence that is being considered for addition; this is to avoid copying the summary dict for every sentence'''
  total = 0.0
  words = set(p_sum.keys() + p_sent.keys())
  for word in words:
    p_word = (p_sum.get(word, 0.0) + p_sent.get(word, 0.0)) / length
    q_word = q[word]
    total += p_word * math.log(p_word/q_word)
  return total
 
def make_unigram_dict(tokens):
  '''Create frequency distribution'''
  freq_dict = {}
  for token in tokens:
    freq_dict[token] = freq_dict.get(token, 0.0) + 1.0
  return freq_dict
def update(sum_dict, sent_dict):
  '''Updates sum_dict with values from sent_dict'''
  for (word, freq) in sent_dict.items():
    sum_dict[word] = sum_dict.get(word, 0.0) + freq

### Our Summarizer ###
#features: (subject to change)
## NER: the number of each type of named entity in the sentence
## topic words: how many topic words in the entire collection are found in the sentence
## sentiment words: the number of each type of sentiment word in the sentence
## sentence position in document
## specificity: the number of words with high/medium/low specificity

def hypernym_distance(word): #From HW4
  '''Finds shortest distance between noun senses of the word and the root hypernym'''
  paths = []
  for s in wn.synsets(word, pos=wn.NOUN):
    paths.extend(s.hypernym_paths())
  paths_greater_than_1 = filter(lambda i: i!= 1, map(len, paths))
  if len(paths_greater_than_1) > 0:
    return min(paths_greater_than_1)
  else:
    return 1

def count_specificities(sentence):
  '''Counts the number of general, medium, and specific, or unspecified entities in the sentence, as well as the average specificity of the non-1 words'''
  specs = [0, 0, 0, 0]
  total_spec = 0.0
  high_threshold = 8  #decided by looking at stats of dev input
  med_threshold = 4
  for word in word_tokenize(sentence):
    if len(wn.synsets(word, pos=wn.NOUN)) != 0:
      spec = hypernym_distance(word)
      if spec == 1: specs[3] += 1
      elif spec >  high_threshold: specs[2] += 1
      elif spec > med_threshold: specs[1] += 1
      else: specs[0] += 1
      if spec != 1: total_spec += spec
  div = specs[0] + specs[1] + specs[2]
  if div == 0: specs.append(0)
  else: specs.append(total_spec/div)
  return specs


#### feature: topic words ####
def load_topic_words(topic_file):
    dct = {}
    f = open(topic_file, 'r')
    content = f.read()
    f.close()
    lines = content.split('\n')
    for line in lines:
        tokens = line.split()
        if(tokens != []):
            dct[tokens[0]] = float(tokens[1])
    return dct

def get_top_n_topic_words(topic_words_dict, n):
    sorted_dict = sorted(topic_words_dict.iteritems(), key=lambda item: -item[1])
    sorted_list = [ x[0] for x in sorted_dict[:n] ]
    return sorted_list

def write_config_files(dev_path):
  dirs = get_sub_directories(dev_path)
  ts_files = []
  config_files = []
  if not os.path.isdir(TW_DIR): os.mkdir(TW_DIR)
  for directory in dirs:
    ts_file = CURRENT_DIR + TW_DIR + directory + '.ts'
    if not os.path.exists(ts_file):
        config_file = CURRENT_DIR + TW_DIR + 'config_' + directory
        content = 'stopFilePath = stoplist-smart-sys.txt\n'
        content += 'performStemming = N\n'
        content += 'backgroundCorpusFreqCounts = bgCounts-Giga.txt\n'
        content += 'topicWordCutoff = 0.1\n'
        content += 'inputDir = ' + dev_path + directory + '\n'
        content += 'outputFile = ' + ts_file + '\n'
        with open(config_file, 'wa') as f:
            f.write(content)
        config_files.append(config_file)
    ts_files.append(ts_file)
  return config_files, ts_files  

def gen_ts_files(dev_path):
    config_files, ts_files = write_config_files(dev_path)
    os.chdir("/home1/c/cis530/hw4/TopicWords-v2/")
    for config_file in config_files:
        os.system("java -Xmx1000m TopicSignatures " + config_file)
    os.chdir(CURRENT_DIR)
    return ts_files 

def get_top_topic_words(ts_file, n):
    dct = load_topic_words(ts_file)
    return  get_top_n_topic_words(dct, n)

def count_sentence_topic_words(sentence):
    words = word_tokenize(sentence)
    intersect = set(words).intersection(set(TOPIC_WORDS))
    return len(intersect)


#### feature: sentence position ####
def build_sentence_position_dict(directory):
    pos_dict = {}
    files = get_all_files(directory)
    for f in files:
        sents = load_file_sentences(directory +'/'+f)
#        for i in range(len(sents)): pos_dict[sents[i]] = i
        for i in range(len(sents)): 
            if i == 0: pos_dict[sents[i]] = 3
            elif i == 1: pos_dict[sents[i]] = 2
            else: pos_dict[sents[i]] = .5
    global POSITION_DICT
    POSITION_DICT = pos_dict

def get_sentence_position(sentence):
    sentence = sentence.lower()
    return POSITION_DICT[sentence]


def score_sentence(sentence):
  '''Score sentence based on topic word count and average specificity'''
  spec = count_specificities(sentence)
  ts_count = count_sentence_topic_words(sentence)
  position = get_sentence_position(sentence)
  if ts_count == 0: ts_count = 0.5
#  return spec[-1] * position / ts_count
  return spec[-1] / (position * ts_count)

def test_helper(sentences):
  '''Summarize using sentence specificity and topic words'''
  summary = []
  tfidf_dict = make_tfidf_dict(sentences)
  scores = []
  for sentence in sentences:
    scores.append(score_sentence(sentence))

  pq = PriorityQueue()
  for pair in zip(scores, sentences):
    pq.put(pair)
  while summary_length(summary) <= 100 and not pq.empty():
    score, next_sentence = pq.get()
    if is_valid(next_sentence, summary, tfidf_dict):
      summary.append(next_sentence)
  return "\n".join(summary)

def summarize(input_collection, output_folder, method):
  '''Creates summaries of input_collection documents using specified method'''
  '''1 = TF*IDF, 2 = LexRank, 3 = KL, 4 = our summarizer'''
  if not input_collection.endswith('/'): input_collection += '/'
  if not output_folder.endswith('/'): output_folder += '/'
  if method == 4: ts_files = gen_ts_files(input_collection)
  dir_list = get_sub_directories(input_collection)
  for i in range(len(dir_list)):
    directory = dir_list[i]
    sentences = load_collection_sentences(input_collection + directory)
    if (method == 1): summary = gen_TFIDF_summary(sentences)
    elif (method == 2) : summary = lex_sum_helper(input_collection + directory)
    elif (method == 3) : summary = gen_KL_summary(sentences)
    elif (method == 4) : 
        global TOPIC_WORDS
        TOPIC_WORDS = get_top_topic_words(ts_files[i], 20)
        build_sentence_position_dict(input_collection + directory)
        summary = test_helper(sentences)
    else : summary = ""
    output = output_folder + gen_output_filename(directory)
    write_to_file(output, summary)

