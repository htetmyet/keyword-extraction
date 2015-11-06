from refo import finditer, Predicate, Plus
from collections import Counter
import nltk
import re
import copy

def get_title(text):
    pre_title = text.splitlines()[0]
    return pre_title

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
    
    test_file = open('dataset/macbook_pro.txt', 'r')
    rawtext = test_file.read()
    #stopwords = open ('smartstoplist.txt', 'r').read()
    
    #Extract title from text
    title = get_title(rawtext)

    #Get paragraph without title
    para_list = rawtext.splitlines()[1:] #in list
    para_string = ''.join(para_list) #convert to string
    
    #Prettify paragraph
    prettify_txt = re.sub(r'[^\w.]', ' ', para_string)
    
    #Tokenizing & POS Tagging
    token_txt = nltk.sent_tokenize(prettify_txt) #Line Segment
    num_sent = len(token_txt) #Number of sentences
    token_word = [nltk.word_tokenize(sent) for sent in token_txt]
    pos_tag = [nltk.pos_tag(sent) for sent in token_word]

    print title
    print "Sentence: ", num_sent
    
    #Chunk and print NP
    get_nouns = [[Word(*x) for x in sent] for sent in pos_tag]

    rule_0 = W(pos = "NN") | W(pos = "NNP")
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


    NP_bi_gram_set = (rule_1)|(rule_2)|(rule_3)|(rule_4)|(rule_5)|(rule_6)|(rule_7)|(rule_8)|(rule_9)|(rule_10)|(rule_11)|(rule_12)|(rule_13)|(rule_14)|(rule_15)|(rule_16)
    NP_tri_gram_set = (rule_17)|(rule_18)|(rule_19)|(rule_20)|(rule_21)|(rule_22)|(rule_23)|(rule_24)|(rule_25)|(rule_26)|(rule_27)|(rule_28)
    NP_quard_gram_set = (rule_29)|(rule_30)

    #Rule set function
    get_uni_gram = (rule_0)
    get_bi_gram = NP_bi_gram_set
    get_tri_gram = NP_tri_gram_set
    get_quard_gram = NP_quard_gram_set

    bag_of_NP = []
    #Get uni-grams NP
    print "UNIGRAM -->"
    for k, s in enumerate(get_nouns):
        for match in finditer(get_tri_gram, s):
            x, y = match.span() #the match spans x to y inside the sentence s
            print pos_tag[k][x:y]
            bag_of_NP += pos_tag[k][x:y]
        
    print "Term Frequency for each:"
    fdist = nltk.FreqDist(bag_of_NP)
    for word in fdist:
        print '%s->%d' % (word, fdist[word])
    print "\n\n"    

if __name__ == '__main__':
    main()

