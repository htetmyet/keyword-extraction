from refo import finditer, Predicate, Plus
from collections import Counter

import math, nltk, re, copy, glob

def remov_stopword(text):
    stopwords = open ('nothesmartstoplist.txt', 'r').read().splitlines()
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text

def count_total_corpus():
    tot_corpus =  len(glob.glob1("traindata","doc*.txt"))
    return tot_corpus

def get_title(content):
    pre_title = content.splitlines()[0]
    return pre_title

def get_first_sen(content):
    get_first = content.splitlines()[1]
    return get_first

def main():
    get_total = count_total_corpus()
    
    while (get_total > 0):
        n_files = str(get_total)
        get_doc = open('traindata/doc'+n_files+'.txt', 'r')
        raw_doc = get_doc.read()

        ##Extract title##
        title = get_title(raw_doc)
        ##Extract First Sentence##
        fir_sen = get_first_sen(raw_doc)

        get_content = raw_doc.splitlines()[1:] #List form
        content_str = ''.join(get_content) #content in String format
        prettify_txt = re.sub(r'[^\w.]',' ', content_str)
        mod_txt = remov_stopword(prettify_txt)
        token_txt = nltk.sent_tokenize(mod_txt)
        print token_txt
        #######################
        get_total -= 1
        
if __name__ == '__main__':
    main()
