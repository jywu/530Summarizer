# Emily Boggs: emboggs@seas.upenn.edu
# Jingyi Wu: wujingyi@seas.upenn.edu

import itertools
from operator import itemgetter
from Queue import PriorityQueue
import numpy as np
from numpy import linalg as LA
import math, os, operator, subprocess, re
import xml.etree.ElementTree as XMLTree
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
NPQA = '/home1/c/cis530/hw3/mpqa-lexicon/subjclueslen1-HLTEMNLP05.tff'
POSITION_DICT = dict()
TS_FILES = []
STOP = set(stopwords.words('english'))
SVM_MODEL = 'dev_svm_model'
REDUNDANCY_THRESHOLD = 0.8
CURRENT_DIR = os.getcwd()

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
    dir_list = get_sub_directories(input_collection)
    for directory in dir_list:
#        if directory == 'dev_00':
        print directory
        # generate input and output paths
        dir_path = input_collection + "/" + directory
        output_file = output_folder + "/" + gen_output_filename(directory)
        # create summary and write to file
        summary = lex_sum_helper(dir_path)
        write_to_file(output_file, summary)
    
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

#### feature: NER ####
def make_corenlp_files(input_collection):
  filelist = "__temp_nlpfiles"
  f = open("filelist", 'w')
  for dr in get_sub_directories(input_collection):
    for doc in get_all_files(dr):
      f.write(input_collection + dr + "/" + doc + "\n")
  f.close()
  os.mkdir("coreNLP_files")
  
  fn_call = 'java -cp /home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/stanford-corenlp-2012-07-09.jar:/home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/stanford-corenlp-2012-07-06-models.jar:/home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/xom.jar:/home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/joda-time.jar -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse -filelist FLIST -outputDirectory OPDIR'
  args = fn_call.split(" ")
  args[8] = filelist
  args[10] = output_dir
  subprocess.call(args)
  os.remove(filelist)

def map_named_entities(xml_collection):
  '''Returns list of lists of NER results'''
  NER_list = []
  entity_types = ['ORGANIZATION', 'PERSON', 'LOCATION', 'MONEY', 'DATE']
  for xml_file in get_all_files(xml_collection):
    root = XMLTree.parse(xml_collection + xml_file).getroot()
    for sent in root.iter("sentence"):
      e_list = [0] * 5
      for ner in sent.iter('NER'):
        if ner.text in entity_types: e_list[entity_types.index(ner.text)] += 1
      NER_list.append(e_list)
  return NER_list

#def count_named_entities(sentence):
#  '''Runs Stanford-NER to count the number of each type of named entity in the sentence'''
#  temp_file = "__tempfile__"
#  write_to_file(temp_file, sentence)
#  arg_string = "java -mx500m -cp /project/cis/nlp/tools/stanford-ner/stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier /project/cis/nlp/tools/stanford-ner/classifiers/ner-eng-ie.crf-3-all2006-distsim.ser.gz -textFile " + temp_file + " -outputFormat inlineXML"
#  arg_list = arg_string.split(" ")
#  error_output = open("/dev/null")
#  proc = subprocess.Popen(arg_list, stdout=subprocess.PIPE, stderr=error_output)
#  (output, err) = proc.communicate()
#  error_output.close()
#  os.remove(temp_file)
#  return map(lambda tag: output.count(tag), ["<PERSON>", "<ORGANIZATION>", "<LOCATION>"])

#### feature: specificity ####
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
      if spec != 1: total += spec
  specs.append(total/(specs[0] + specs[1] + specs[2]))
  return specs

#### feature: sentence length ####
def count_sentence_lengths(sentence):
    return [len(word_tokenize(sentence))]

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
  for directory in dirs:
    ts_file = CURRENT_DIR + '/' + directory + '.ts'
    if os.path.exists(ts_file):
        config_file = CURRENT_DIR + '/config_' + directory
        content = 'stopFilePath = stoplist-smart-sys.txt\n'
        content += 'performStemming = N\n'
        content += 'backgroundCorpusFreqCounts = bgCounts-Giga.txt\n'
        content += 'topicWordCutoff = 0.1\n'
        content += 'inputDir = ' + dev_path + '/' + directory + '\n'
        content += 'outputFile = ' + ts_file + '\n'
        with open(config_file, 'wa') as f:
            f.write(content)
        config_files.append(config_file)
    ts_files.append(ts_file)
  return config_files, ts_files  

def gen_ts_files(dev_path):
    config_files, ts_files = write_config_files(dev_path)
    if config_files == []:
        for config_file in config_files:
            os.chdir("/home1/c/cis530/hw4/TopicWords-v2/")
            os.system("java -Xmx1000m TopicSignatures " + config_file)
    return ts_files 

def get_top_topic_words(ts_file, n):
    dct = load_topic_words(ts_file)
    return  get_top_n_topic_words(dct, n)

def count_sentence_topic_words(sentence):
    words = word_tokenize(sentence)
    intersect = set(words).intersection(set(TOPIC_WORDS))
    return [len(intersect)]

#### feature: negative/positive (from hw4 modified) ####
def get_mpqa_lexicon():
  mpqa = {}
  with open(NPQA, 'r') as f:
    for line in f:
      properties = to_map(line)
      word = properties['word1']
      subj_type = properties['type']
      polarity = properties['priorpolarity']
      tuples = mpqa.get(word, [])
      tuples.append((subj_type, polarity))
      mpqa[word] = tuples
  return mpqa

def to_map(line):
  properties = {}
  tokens = line.split()
  for token in tokens:
    try:
      key, value = token.split('=')
    except ValueError:
      continue
    properties[key] = value
  return properties

def get_mpqa_features(sentence, dictionary):
  words = word_tokenize(sentence)
  counts = {}
  for word in words:
    polarities = dictionary.get(word)
    if polarities == None:
      continue
    for polarity in polarities:
      counts[polarity[1]] = counts.get(polarity[1], 0) + 1
  return [
  counts.get('positive', 0),
  counts.get('negative', 0),
  counts.get('neutral', 0)
  ]

#### feature: sentence position ####
def build_sentence_position_dict(directory):
    dct = dict()
    files = get_all_files(directory)
    for f in files:
        sents = load_file_sentences(directory +'/'+f)
        for i in range(len(sents)):
            if i == 0:
                dct[sents[0]] = 3
            elif i == 1:
                dct[sents[1]] = 1
            else:
                dct[sents[i]] = 0
    global POSITION_DICT
    POSITION_DICT = dct

def get_sentence_position(sentence):
    sentence = sentence.lower()
    return [POSITION_DICT[sentence]]

#### add all features, train data, generate summary ####
def write_feature_file(sentence_list, feature_file, label_list=None):
  '''Writes SVM file containing feature vectors for sentences'''
  ner_features = map_named_entities("coreNLP_files")
  f = open(feature_file, 'w')
  for i in range(len(sentence_list)):
    print "writing to feature file"
    features = get_features(sentence_list[i]).append(ner_features[i])
    if not label_list: f.write("0") #label
    else: f.write(label_list[i])
    f.write(" qid:1")
    for j in range(len(features)):
      if features[j] != 0: f.write(" " + str(j + 1) + ":" + str(features[j]))
    f.write("\n")
  f.close()
      
    
def get_features(sentence):
  features = []
  # features.extend(count_specificities(sentence))
  features.extend(count_sentence_lengths(sentence))
  features.extend(count_sentence_topic_words(sentence))
  features.extend(get_mpqa_features(sentence, MPQA_DICT))
  # features.extend(get_sentence_position(sentence))
    #TODO add features
  return features

def get_rankings(sentences):
  '''Gets ranking of sentences in this collection'''
  feature_file = "__temp_features"
  predict_file = "__temp_predict"
  write_feature_file(sentences, feature_file)

  if not os.path.isfile(SVM_MODEL): train_svm()
  subprocess.call(["/project/cis/nlp/tools/svmRank/svm_rank_classify", feature_file, SVM_MODEL, predict_file]) 
  
  p = open(predict_file, 'r')
  predictions = map(lambda x: float(x.strip("\n")), p.readlines())
  p.close()
  pq = PriorityQueue()
  for pair in zip(predictions, sentences):
    pq.put(pair)
  return pq

def ml_summary(sentences):
  '''Uses trained classifier to extract summary'''
  pq = get_rankings(sentences)
  summary = [] 
  tf_idf_dict = make_tfidf_dict(sentences)
  while summary_length(summary) <= 100 and not pq.empty():
    rank, next_sentence = pq.get()
    if is_valid(next_sentence, summary, tfidf_dict):
      summary.append(next_sentence)
  return "\n".join(summary)

def train_svm():
  '''Trains SVM-Rank model using sentences rankings based under perplixity from model-summary LMs'''
  lm_data = "__temp_srilmdata"
  lm_file = "__temp_srilmtrain"
  svm_data = "__temp_svmdata"
  labels = []
  sentences = []

  model_files = get_all_files(DEV_MODELS)
  ts_files = gen_ts_files(DEV)
  dev_directories = get_sub_directories(DEV)
  for i in range(len(dev_directories)):
    dev_set = dev_directories[i]
    print dev_set
    models = filter(lambda f: f.startswith(dev_set), model_files)
    write_model_file(models, lm_data)
    subprocess.call(["/home1/c/cis530/hw2/srilm/ngram-count", "-text", lm_data, "-lm", lm_file])
    for dev_doc in get_all_files(DEV + dev_set):
      (sents, ppls) = get_sentences_and_ppl(DEV + dev_set + "/" + dev_doc, lm_file)
      sentences.extend(sents)
      labels.extend(ppls)
  write_feature_file(sentences, svm_data,labels)
  subprocess.call(["/project/cis/nlp/tools/svmRank/svm_rank_learn", "-c", "20",svm_data, SVM_MODEL]) #TODO c value?
  os.remove(lm_data); os.remove(lm_file); os.remove(svm_data)
    
def write_model_file(models, filename):
  '''Writes all models to same file for SRILM processing'''
  output = open(filename, 'w')
  for model in models:
    model_sents = load_file_sentences(DEV_MODELS + model)
    for sent in model_sents: output.write(" ".join(sent) + '\n')
  output.close()

def get_sentences_and_ppl(document, lm_file):
  '''Uses LM to calculate perplexities of sentences in file; returns tuple of sentences and ppls'''
  args = ['/home1/c/cis530/hw2/srilm/ngram', '-lm', lm_file, '-ppl', document, "-debug", "1"]
  proc = subprocess.Popen(args, stdout=subprocess.PIPE)
  (output,err) = proc.communicate()
  output_lines = output.split("\n")
  sents = []
  ppls = []
  i = 0
  while i < len(output_lines) - 4:
    if i % 4 == 0: sents.append(output_lines[i].strip())
    if i % 4 == 2: ppls.append(re.search("ppl= (\S+)", output_lines[i]).group(1))
    i += 1
  return sents, ppls


def xmlfiles_exist(input_collection):
  '''Checks that all necessary XML files from CoreNLP exist'''
  xml_dir = "coreNLP_files"
  if not os.path.isdir(xml_dir): return False
  for dr in get_sub_directories(input_collection):
    for doc in get_all_files(input_collection + dr):
      if not os.path.isfile(xml_dir + "/" + doc + ".xml") : return False
  return True

def summarize(input_collection, output_folder, method):
  if not input_collection.endswith('/'): input_collection += '/'
  if not output_folder.endswith('/'): output_folder += '/'
  # preprocess for ml_summary
  if method == 4:
    ts_files = gen_ts_files(DEV)
    global MPQA_DICT 
    MPQA_DICT = get_mpqa_lexicon()
    if not xmlfiles_exist(input_collection):
      make_corenlp_files(input_collection)
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
        print POSITION_DICT
        summary = ml_summary(sentences)
    else : summary = ""
    output = output_folder + gen_output_filename(directory)
    write_to_file(output, summary)

#summarize(DEV, '../ours', 4)
