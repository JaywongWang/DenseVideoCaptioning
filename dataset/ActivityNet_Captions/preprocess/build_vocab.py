"""
This script is to build word dictionary for input data
Afterwards build train/val/test id and encoded sentence data
"""
import os
import json
import numpy as np
import h5py
import re
import time
import sys
#from importlib import reload
from collections import OrderedDict

reload(sys)
sys.setdefaultencoding('utf-8')

unk_word = '<UNK>'
start_word = '<START>'
end_word = '<END>'
punctuations = [':','!','?','.',';','(',')','-','_','\n']
count_threshold = 4 # remove tail words


## sentence preprocessing: add start word and end word, remove all punctuations, lower case
def preprocess_caption(sentence):
    # remove all punctuations
    for p in punctuations:
        sentence = sentence.replace(p,'')   # some notation should be kept in the vocab
    sentence = sentence.replace(',', ' , ') # keep , notation to inforce sentence smoothness
    
    return sentence.lower().split()
    
## encode sentence: use id to represent a word, a sentence is encoded as a list of id
def encode_caption(sentence, vocab):
    tokens = preprocess_caption(sentence)
    tokens_id = [vocab[start_word]] + [vocab.get(x, vocab[unk_word]) for x in tokens] + [vocab[end_word]]
    
    return tokens_id

 # build caption encoded data
def build_encoded_caption(data, vocab, save_json):

    encoded_captions = [[encode_caption(caption, vocab) for caption in data[vid]['sentences']] for vid in data.keys()]
    with open(save_json, 'w') as fid:
        json.dump(encoded_captions, fid)

## build vocabulary: generate vocab .txt file and .json file 
def build_vocabulary(train_data, out_vocab_file, out_encoded_vocab_file):
    # first get word frequency
    vocab_freq = {}
    video_ids = train_data.keys()
    sentence_count = 0
    for video_id in video_ids:
        sentences = train_data[video_id]['sentences']
        for sentence in sentences:
            sentence_count += 1
            tokens = preprocess_caption(sentence)
            for token in tokens:
                if token in vocab_freq.keys():
                    vocab_freq[token] = vocab_freq[token] + 1
                else:
                    vocab_freq[token] = 1
    
    # remove words with low frequency 
    print('Dictionary size: %d'%len(vocab_freq)) 
    vocabs = vocab_freq.copy()
    for word in vocab_freq.keys():
        if vocabs[word] < count_threshold:
            vocabs.pop(word)

    # add special word: 'UNK' (to represent unknown word), 'START', and 'END' word
    vocabs[unk_word] = len(vocab_freq) - len(vocabs)
    vocabs[start_word] = sentence_count
    vocabs[end_word] = vocabs[start_word]
    print('After removing tail words, dictionary size: %d'%len(vocabs))
    
    # sort by frequency 
    vocabs_sort = OrderedDict(sorted(vocabs.items(), key=lambda t:t[1], reverse=True))
    
    # write vocab frequency file
    vocab_freq_fid = open(out_vocab_file, 'w')
    for word in vocabs_sort.keys():
        vocab_freq_fid.write(word + ' ' + str(vocabs_sort[word]) + '\n')
    vocab_freq_fid.close()
    
    # encode 
    encoded_vocab = {}
    id = 0
    for word in vocabs_sort.keys():
        encoded_vocab[word] = id
        id = id + 1
    # write encoded vocab
    print('Saving encoded dictionary ...')
    with open(out_encoded_vocab_file, 'w') as encoded_vocab_fid:
        json.dump(encoded_vocab, encoded_vocab_fid)
    
    return encoded_vocab
 

vocabulary_file = 'vocabulary.txt'
encoded_vocabulary_file = 'word2id.json'

# time
start_time = time.time()

print('Loading json data ...')
train_data = json.load(open('../train.json'))
val_data = json.load(open('../val_1.json'))
test_data = json.load(open('../val_2.json'))

print('Writing id file ...')
train_ids = [vid+'\n' for vid in train_data.keys()]
val_ids = [vid+'\n' for vid in val_data.keys()]
test_ids = [vid+'\n' for vid in test_data.keys()]
with open('train/ids.txt', 'w') as fid:
    fid.writelines(train_ids)
with open('val/ids.txt', 'w') as fid:
    fid.writelines(val_ids)
with open('test/ids.txt', 'w') as fid:
    fid.writelines(test_ids)


print('Building vocabulary ...')
encoded_vocab = build_vocabulary(train_data, vocabulary_file, encoded_vocabulary_file)
print('Done.')


print('Saving encoded dictionary ...')
with open(encoded_vocabulary_file, 'w') as encoded_vocab_fid:
    json.dump(encoded_vocab, encoded_vocab_fid)

print('Encoding captions ...')
build_encoded_caption(train_data, encoded_vocab, 'train/encoded_sentences.json')
build_encoded_caption(val_data, encoded_vocab, 'val/encoded_sentences.json')
build_encoded_caption(test_data, encoded_vocab, 'test/encoded_sentences.json')



end_time = time.time()

print('Total running time: %f seconds.'%(end_time - start_time))