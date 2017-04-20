# -*- coding: utf-8 -*-
"""
Created on Tue Apr 04 16:25:18 2017

@author: Administrator
"""
import glob, random, os, re, nltk
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from decimal import*
from collections import Counter
from textblob import TextBlob

def norm_pos(content):
    get_max = max(content, key=itemgetter(1))[1]
    
    new_str = []
    for each in content:
        get_norm = Decimal(each[1])/Decimal(get_max) 
        new_pos = each[0], get_norm
        new_str.append(new_pos)
    
    return new_str
        
    
def get_cont(doc_name):
    file_path = 'dataset/'
    get_cont = open(file_path+doc_name, 'r')
    raw_txt = get_cont.read()
    
    return raw_txt

 	
def findtags(tag_prefix, tagged_text):
    cfd = nltk.ConditionalFreqDist((tag, word) for (word, tag) in tagged_text
                                  if tag.startswith(tag_prefix))
    return dict((tag, cfd[tag].most_common(5)) for tag in cfd.conditions())
    
def main():
    DIR = 'dataset/'
    data_list = []
    num_files = len(glob.glob1(DIR,"*.txt"))
    print 'Total ',num_files, ' files.'

    get_files = []
    cnt = 0
    num_of_turn = 8
    while (len(get_files) < num_of_turn):
        temp_file = random.choice([x for x in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, x))])
        if temp_file not in get_files:
            get_files.append(temp_file)
            cnt +=1
    
    get_file = raw_input('Enter file name: ')
    get_cont = open(get_file, 'r')
    get_txt = get_cont.read()
    prettify_txt = re.sub(r'[^\w.]', ' ', get_txt)
    token_txt = nltk.sent_tokenize(prettify_txt)
    tokens = nltk.word_tokenize(token_txt)
    tagged = nltk.pos_tag(tokens)
    print tagged    
    
    '''
    for each in get_files:
        get_txt = get_cont(each)
        prettify_txt = re.sub(r'[^\w.]', ' ', get_txt)
        token_txt = nltk.sent_tokenize(prettify_txt)
        ##tokens = nltk.word_tokenize(token_txt)
        token_word = [nltk.word_tokenize(sent) for sent in token_txt]
        pos_tag = [nltk.pos_tag(sent) for sent in token_word]
        
        tagdict = findtags('NN', pos_tag)                
        print tagdict
        ##get_blob = TextBlob(prettify_txt)
        ##print get_blob.noun_phrases '''
        
        
        
if __name__ == '__main__':
    main()
    