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
def get_last_sen(content):
    get_last = content.splitlines()[-1]
    return get_last

class Word(object):
    def __init__(self, token, pos):
        self.token = token
        self.pos = pos

class W(Predicate):
    def __init__(self, token = ".*", pos = ".*"):
        self.token = re.compile(token + "$")
        self.pos = re.compile(pos + "$")
        super(W, self).__init__(self.match)
    def match(self,word):
        m1 = self.token.match(word.token)
        m2 = self.pos.match(word.pos)
        return m1 and m2
    
def main():
    get_total = count_total_corpus()
    count = 0
    while (count < get_total):
        n_files = str(count+1)
        get_doc = open('traindata/doc'+n_files+'.txt', 'r')
        raw_doc = get_doc.read()

        ##Extract title##
        title = get_title(raw_doc)
        ##Extract First&Last Sentence##
        fir_sen = get_first_sen(raw_doc)
        last_sen = get_last_sen(raw_doc)
        get_last = last_sen.split(',')
        get_length = len(get_last)
        #### KEYWORD SECTION ####
        x=0
<<<<<<< HEAD
        for word in get_last:
            print word[x]
=======
        key_unigram = ''
        key_bigram = ''
        key_trigram = ''
        key_fourgram = ''
        key_unknown = ''
>>>>>>> origin/master
        
        while (x<get_length):
            get_len = len(get_last[x].split())
            if (get_len == 1):
                key_unigram += get_last[x]+','
            elif (get_len == 2):
                key_bigram += get_last[x]+','
            elif (get_len == 3):
                key_trigram += get_last[x]+','
            elif (get_len == 4):
                key_fourgram += get_last[x]+','
            else:
                key_unknown += get_last[x]+','
            x += 1
        print type(key_unigram)
            
        get_content = raw_doc.splitlines()[1:] #List form
        content_str = ''.join(get_content) #content in String format
        prettify_txt = re.sub(r'[^\w.]',' ', content_str)
        mod_txt = remov_stopword(prettify_txt)
        token_txt = nltk.sent_tokenize(mod_txt)
        ##Number of Sentence: len(token_txt)##
        token_word = [nltk.word_tokenize(sent) for sent in token_txt]
        pos_tag = [nltk.pos_tag(sent) for sent in token_word]

        print title
        print '\n'
        ##Chunking and printing  NP##
        get_nouns = [[Word(*x) for x in sent] for sent in pos_tag]
        ##NNP Rules##
        rule_0 = W(pos = "NNS")| W(pos = "NNS")| W(pos = "NN") | W(pos = "NNP")
        rule_05 = W(pos = "NNP") + W(pos = "NNS")
        rule_1 = W(pos = "WP$") + W(pos = "NNS")
        rule_2 = W(pos = "CD") + W(pos = "NNS")
        rule_3 = W(pos = "NN") + W(pos = "NN")
        rule_4 = W(pos = "NN") + W(pos = "NNS")
        rule_5 = W(pos = "NNP") + W(pos = "CD")
        rule_6 = W(pos = "NNP") + W(pos = "NNP")
        rule_7 = W(pos = "NNP") + W(pos = "NNPS")
        rule_8 = W(pos = "NNP") + W(pos = "NN")
        rule_9 = W(pos = "NNP") + W(pos = "VBZ")
        rule_10 = W(pos = "DT") + W(pos = "NNS")
        rule_11 = W(pos = "DT") + W(pos = "NN")
        rule_12 = W(pos = "DT") + W(pos = "NNP")
        rule_13 = W(pos = "JJ") + W(pos = "NN")
        rule_14 = W(pos = "JJ") + W(pos = "NNS")
        rule_15 = W(pos = "PRP$") + W(pos = "NNS")
        rule_16 = W(pos = "PRP$") + W(pos = "NN")
        rule_02 = W(pos = "NN") + W(pos = "NN") + W(pos = "NN")
        rule_17 = W(pos = "NN") + W(pos = "NNS") + W(pos = "NN")
        rule_18 = W(pos = "NNP") + W(pos = "NNP") + W(pos = "NNP")
        rule_19 = W(pos = "JJ") + W(pos = "NN") + W(pos = "NNS")
        rule_20 = W(pos = "PRP$") + W(pos = "NN") + W(pos = "NN")
        rule_21 = W(pos = "DT") + W(pos = "JJ") + W(pos = "NN")
        rule_22 = W(pos = "DT") + W(pos = "CD") + W(pos = "NNS")
        rule_23 = W(pos = "DT") + W(pos = "VBG") + W(pos = "NN")
        rule_24 = W(pos = "DT") + W(pos = "NN") + W(pos = "NN")
        rule_25 = W(pos = "NNP") + W(pos = "NNP") + W(pos = "VBZ")
        rule_26 = W(pos = "DT") + W(pos = "NNP") + W(pos = "NN")
        rule_27 = W(pos = "DT") + W(pos = "NNP") + W(pos = "NNP")
        rule_28 = W(pos = "DT") + W(pos = "JJ") + W(pos = "NN")
        rule_29 = W(pos = "DT") + W(pos = "NNP") + W(pos = "NNP") + W(pos = "NNP")
        rule_30 = W(pos = "DT") + W(pos = "NNP") + W(pos = "NN") + W(pos = "NN") 

        NP_bi_gram_set = (rule_05)|(rule_1)|(rule_2)|(rule_3)|(rule_4)|(rule_5)|(rule_6)|(rule_7)|(rule_8)|(rule_9)|(rule_10)|(rule_11)|(rule_12)|(rule_13)|(rule_14)|(rule_15)|(rule_16)
        NP_tri_gram_set = (rule_02)|(rule_17)|(rule_18)|(rule_19)|(rule_20)|(rule_21)|(rule_22)|(rule_23)|(rule_24)|(rule_25)|(rule_26)|(rule_27)|(rule_28)
        NP_quard_gram_set = (rule_29)|(rule_30)

        #Rule set function
        get_uni_gram = (rule_0)
        get_bi_gram = NP_bi_gram_set
        get_tri_gram = NP_tri_gram_set
        get_quard_gram = NP_quard_gram_set

        bag_of_NP = []
        bag_of_biNP = []
        bag_of_triNP = []
        bag_of_fourNP = []
        total__tfidf = 0
        #######################
        for k, s in enumerate(get_nouns):
            for match in finditer(get_uni_gram, s):
                x, y = match.span() #the match spans x to y inside the sentence s
                ##print pos_tag[k][x:y]
                bag_of_NP += pos_tag[k][x:y]
        for k, s in enumerate(get_nouns):
            for match in finditer(get_bi_gram, s):
                x, y = match.span()
                ##print pos_tag[k][x:y]
                bag_of_biNP += pos_tag[k][x:y]
        for k, s in enumerate(get_nouns):
            for match in finditer(get_tri_gram, s):
                x, y = match.span()
                ##print pos_tag[k][x:y]
                bag_of_triNP += pos_tag[k][x:y]
        for k, s in enumerate(get_nouns):
            for match in finditer(get_quard_gram, s):
                x,y = match.span()
                ##print pos_tag[k][x:y]
                bag_of_fourNP += pos_tag[k][x:y]
        print bag_of_NP
        print bag_of_biNP
        print bag_of_triNP
        print bag_of_fourNP
            
        #######################          
        count += 1
        
if __name__ == '__main__':
    main()
