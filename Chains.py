import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
 
threshold = 0.6 #treshold for wup
jcnTreshold = 0.09 #jcn
pathTeshold = 0.1 #path
brown_ic = wordnet_ic.ic('ic-brown.dat') #load the brown corpus
lexical_chains = [] #empty list to hold all the chains
dictionary = {} #empty dictionart to hold the count of each word encountered

#class Chain 
class Chain(): 
    def __init__(self, words, senses, count = 0):
        self.words = set(words)
        self.senses = set(senses)
        dictionary[words[0]] = 1 #initialize counter
     
    def addWord(self, word):
        
        if(len(self.words.intersection([word])) > 0):
            dictionary[word] += 1
        else:
            dictionary[word] = 1
        
        self.words.add(word)
     
     

    def addSense(self, sense):
     self.senses.add(sense)

    def getWords(self):
     return self.words

    def getSenses(self):
     return self.getSenses

    def incCount(self):
        self.count += 1



def add_word(word):
    maximum = 0 
    maxJCN = 0
    flag = 0
    for chain in lexical_chains: #for all chains that are present
     for synset in wn.synsets(word): #for all synsets of current word
         for sense in chain.senses:  #for all senses of the current word in current element of the current chain
             similarity = sense.wup_similarity(synset) #using wup_similarity
             
             if(similarity >= maximum):
                 if similarity >= threshold:
                     #print word, synset, sense, sense.jcn_similarity(synset, brown_ic)
                     JCN = sense.jcn_similarity(synset, brown_ic) #using jcn_similarity
                     if JCN >= jcnTreshold: 
                         if sense.path_similarity(synset) >= 0.2: #using path similarity
                             if JCN >= maxJCN:
                                 maximum = similarity
                                 maxJCN = JCN
                                 maxChain = chain
                                 flag = 1
    if flag == 1:                                           
        maxChain.addWord(word)
        maxChain.addSense(synset)
        return
              
    lexical_chains.append(Chain([word], wn.synsets(word)))


fileName = raw_input("Enter file path + name, if file name is 'nlp.txt', type 'nlp' \n \n")
fileName += ".txt"
print ("\n\n")
#fileName = "nlp.txt"
File = open(fileName) #open file
lines = File.read() #read all lines 

is_noun = lambda x: True if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS') else False
nouns = [word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(lines)) if is_noun(pos)]  #extract all nouns

for word in nouns:
    add_word(word)

#print all chains
for chain in lexical_chains:
    print ", ".join(str(word + "(" + str(dictionary[word]) + ")") for word in chain.getWords())
