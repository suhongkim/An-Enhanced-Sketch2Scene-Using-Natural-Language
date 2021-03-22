import os, sys, optparse
import tqdm
import pymagnitude
import numpy as np
import re
import copy
import subprocess
import json
import nltk
from collections import Counter

## Stopwords--------------------------------
un_stopwords = ["it's", "she's", 'were', 'because', 'this', 'couldn', 'then', 'how'
, 'd', 'doesn', 'down', 's', 'they', 'she', "needn't", 'wasn', 'haven', 
'between', "wouldn't", 'the', 'ma', "wasn't", 'until', 'my', 'himself', 
"that'll", 'by', 'about', 'in', "aren't", "should've", 'why', 'nor', 
'before', 'when', 'we', 'here', 'only', "couldn't", 'ain', 'no', 'your', 
'will', 'own', 'his', "you'll", 'are', 'and', 'most', 'do', 'now', "isn't", 
'having', 'on', 'her', 'theirs', 'under', 'with', 'to', "mightn't", 'while', 
'its', 'be', 'll', 'don', 'over', 'again', 'their', 'won', 'too', 'during', 
'shan', 'herself', 'has', 'or', 'from', 'ours', 'into', 'our', 'above', 
'wouldn', 'you', 'of', 'so', 't', 'he', 'doing', 'as', 'i', 'can', 'shouldn', 
'have', 'at', 'other', 'hasn', 'more', 'yourselves', 'y', 'yours', 'very', 
'themselves', 'which', 'these', 'being', 'both', 'aren', 'did', 'than', 'needn',
 'for', 'itself', "haven't", 'through', 'weren', 'but', 'once', 'isn', 
 'ourselves', 'didn', 'not', 'yourself', 'mightn', 'after', 've', 'him', 
 'whom', "hasn't", 'a', 'hadn', "shouldn't", "mustn't", 'those', 'off', 
 'each', 'was', "didn't", "you'd", 'where', 'o', 'further', 'below', "shan't", 
 'myself', 'mustn', 'is', 'been', 'just', 'any', 'out', 'that', 'm', 'such', 
 'me', 'same', 'hers', 'some', 'had', 'does', 'against', 'should', "you've", 
 "doesn't", "you're", 'them', 'am', 'if', 'who', 'few', 'what', 'there', 
 "don't", "weren't", "won't", 'an', 'all', 're', 'it', 'up', "hadn't", "'ll"]
nltk_stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
nltk_stopwords = set(nltk_stopwords).union(set(un_stopwords))

def filterWords(words,word_vectors):  
    r = re.compile(r'[^\w\s]') # punctuation
    punc = list(filter(r.search,words))
    words = [w for w in words if w not in punc]
    words = [w for w in words if w not in nltk_stopwords]
    words = [w for w in words if w in word_vectors.keys()]
    return words        

def getContexts(word_vec, train_file, gt_file, context_window=5):
    dev_inputs = []
    with open(train_file, 'rt') as f:
        for line in f:
            dev_inputs.append(line)
            
    ref_data = []
    with open(gt_file, 'rt') as refh:
        ref_data = [str(x).strip() for x in refh.read().splitlines()]
   
    context_dict = {}
    for i in range(len(dev_inputs)):
        gt = ref_data[i].strip().split('\t')
        inp = dev_inputs[i].strip().split('\t')
        index = int(inp[0])
        sentence = inp[1].strip().split()
        keys = gt[1].strip().split()
        
        window = range(max(0,index-context_window), min(len(sentence), index+context_window+1))
        contexts = [sentence[c] for c in window if c != index]
        contexts = filterWords(contexts, word_vec)
        for key in keys:
            if key in context_dict.keys():
                prev_contexts = context_dict[key]
                context_dict[key] = prev_contexts + contexts
            else:
                context_dict[key] = contexts
    return context_dict
    

## Lexicon---------------------------------
def norm_word(word):
    if is_number(word.lower()):
        return '---num---'
    elif re.sub(r'\W+', '', word) == '':
        return '---punc---'
    else:
        return word.lower()

def is_number (in_str):
    try :
        in_str = int(in_str)
        return True
    except:
        return False

def readLexicon(lexicon_file):
    lexicon = {}
    for line in open(lexicon_file, 'r'):
        words = line.lower().strip().split()
        lexicon[norm_word(words[0])] = [norm_word(word) for word in words[1:]]
    return lexicon

## Retrofitting---------------------------------
class Retrofitting:
    def __init__(self, wv_magnitude_file, n_iter=10):
        # word embeddings from magnitude file
        # self.wv = self.readMagnitude(wv_magnitude_file)
        # initialize retrofitted wv from wv
        # self.retro_wv = copy.deepcopy(self.wv) 
        self.retro_wv = None      
        
        self.n_iter = n_iter   

    def retrofitting(self, lexicon, alpha=1, beta=1):   
            new_word_vector = copy.deepcopy(self.retro_wv)
            vocablary = set(new_word_vector.keys())
            common_vocab_lexicon = vocablary.intersection(set(lexicon.keys()))
            for iter in range(self.n_iter):
                for word in common_vocab_lexicon:
                    word_neighbours = set(lexicon[word])
                    word_neighbours_inter = word_neighbours.intersection(vocablary)
                    num_neighnours = len(word_neighbours_inter)
                    if num_neighnours == 0:
                        continue

                    new_vector = alpha * num_neighnours * self.retro_wv[word]
                    for neighbours in word_neighbours_inter:
                        new_vector += beta * new_word_vector[neighbours]

                    new_word_vector[word] = new_vector/((beta+alpha)*num_neighnours)
            # update to the current word_vector
            self.retro_wv = new_word_vector    
         
    def readMagnitude(self, wv_magnitude_file):
        wv = pymagnitude.Magnitude(wv_magnitude_file)
        word_vectors = {}
        for key, vectors in wv:
            word_vectors[key] = np.zeros(len(vectors))
            for index , vector in enumerate(vectors):
                word_vectors[key][index] = vector
        return word_vectors
    
    def writeMagnitude_reduced(self, lexicon_out, wordmap_path):
        with open(wordmap_path, 'r') as j: 
            word_map = json.load(j)
        
        #write into txt 
        new_vector_list = []
        for word in self.retro_wv.keys():
            if word_map.get(word) is None: 
                print(word)
                continue 
            line = word
            for vec in self.retro_wv[word]:
                line = line+" "+str(vec)
            new_vector_list.append(line)

        with open(lexicon_out+".txt",'w') as f:
            for val in new_vector_list:
                f.write(val)
                f.write('\n')    

    def writeMagnitude(self, lexicon_out):   
        #write into txt 
        new_vector_list = []
        for word in self.retro_wv.keys():
            line = word
            for vec in self.retro_wv[word]:
                line = line+" "+str(vec)
            new_vector_list.append(line)

        with open(lexicon_out+".txt",'w') as f:
            for val in new_vector_list:
                f.write(val)
                f.write('\n')
                
        # # generate magnitude
        # subprocess.run(["python3", "-m", "pymagnitude.converter", 
        #                 "-i",lexicon_out+".txt", "-o", lexicon_out], 
        #                check=True)   
        

def create_relation(input_json, output_txt, max_len=5):
    # Read Karpathy JSON
    with open(input_json, 'r') as j:
        data = json.load(j)

    keywords_list = []
    for img in data['images']:
        nn_freq = Counter()
        for c in img['sentences']:
            nn_words = [t[0] for t in nltk.pos_tag(c['tokens']) if t[1] == 'NN']
            nn_freq.update(nn_words)

        keywords = [nn for nn, c in nn_freq.most_common(max_len)]
        keywords_list.append(' '.join(w for w in keywords))
        
    with open(output_txt, 'w') as f:
        f.write('\n'.join(keywords_list))


if __name__ == '__main__':
    # Create input files (along with word map)
    create_relation(input_json='./data/dataset_coco.json',
                    output_txt='./data/coco-retrofitting.txt',
                    max_len=5)

    # # conver w2v into magnitude file 
    wordvecfile = os.path.join('data', 'glove.42B.300d')
    subprocess.run(["python3", "-m", "pymagnitude.converter", 
                    "-i",wordvecfile+".txt", "-o", wordvecfile+".magnitude"], 
                    check=True) 
    
    # retrofitting 
    new_retro_file = os.path.join("data", "glove.42B.300d.retrofit.magnitude")
    if not os.path.exists(new_retro_file):
        # initialize retrofitting class
        retro = Retrofitting(wv_magnitude_file=wordvecfile+".magnitude")
        
        # read ontology files
        coco = readLexicon(os.path.join("data", "coco-retrofitting.txt"))
        retro.retrofitting(coco, alpha=1, beta=1)

        # write the final output into Magnitude format
        retro.writeMagnitude_reduced(new_retro_file, 
                                     'data/coco/WORDMAP_coco_baseline_3.json')
        subprocess.run(["python3", "-m", "pymagnitude.converter", 
                "-i",new_retro_file+".txt", "-o", new_retro_file], 
                check=True)   