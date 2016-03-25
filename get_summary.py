from nltk.corpus import wordnet as wn
from itertools import product
from collections import Counter
import nltk, re, math
    
def extractNouns(title):
    NOUNS = ['NN','NNS','NNP','NNPS','PRP','PRP$']
    nouns = []
    f_nouns = []
    prettify_tit = re.sub(r'[^\w.]', ' ', title)
    text = nltk.word_tokenize(prettify_tit)
    word_tags = nltk.pos_tag(text)
    
    for word_tag in word_tags:
        if word_tag[1] in NOUNS:
            nouns.append(word_tag)
    for n in nouns:
        f_nouns.append(n[0])

    return f_nouns

def get_sent(content):
    num_sen = 1
    sen_length = []
    sen_index = []
    get_sen = nltk.sent_tokenize(content)
    for idx, each in enumerate(get_sen):
        get_len = len(each.split())
        sen_length.append(get_len)
        idx+= num_sen
        sen_index.append(idx)
        
    sent_info = zip(get_sen, sen_index, sen_length)
    
    return sent_info

def chk_hypo(n1, n2):
    chk_res =''
    for s in wn.synsets(n1):
        hypos = s.hyponyms()
        for h in hypos:
            if n2 in h.name():
                chk_res = '1'
            else:
                chk_res = '0'
    return chk_res

def chk_hyper(n1, n2):
    chk_res = ''
    for s in wn.synsets(n1):
        hypers = s.hypernyms()
        for h in hypers:
            if n2 in h.name():
                chk_res = '1'
            else:
                chk_res = '0'
    return chk_res

def chk_synset(n1, n2):
    chk_res = ''
    for s in wn.synsets(n1):
        lemmas = s.lemmas()
        for l in lemmas:
            if n2 in l.name():
                chk_res = '1'
            else:
                chk_res = '0'
    return chk_res

def find_chain(all_nouns, key_nn):
    chain_train = []
    for each in key_nn:
        ##make chain
        get_num = all_nouns.count(each)
        temp_chain = []
        ##print each
        ##print get_num
        for n in all_nouns:
            if each == n:
               temp_chain.append(n)

            elif (chk_hypo(each, n) == '1'):
                temp_chain.append(n)

            elif (chk_hyper(each, n) == '1'):
                temp_chain.append(n)

            elif (chk_synset(each, n) == '1'):
                temp_chain.append(n)
            else:
                pass
        chain_train.append(temp_chain)
    return chain_train

def score_chain(chain, key_nouns, rel_nouns):
    
    num_chain = len(chain)##Num of Chain
    store_scr = []
    for each in chain:
        chain_scr = ''
        length_chain = len(each)#Length of Chain
        #1-(dist_feat/length_chain) float values
        freq_item = Counter(each).items()#Frequency of each item
        disnt_occ = max(freq_item)##Distinct Occurrence
        dist_occ_scr = disnt_occ[1]
        repetit_scr = float ( 1 - (dist_occ_scr/length_chain))
        chain_scr = length_chain * repetit_scr
        
        ##print chain_scr
        for i in each:
            for w,v in key_nouns:
                if (i == w) and (v == 1):
                    chain_scr += 1
                else:
                    pass
            for w,v in rel_nouns:
                if (i == w) and (v == 0.5):
                    chain_scr += 0.5
                else:
                    pass
        store_scr.append(chain_scr)
    get_the_chain = zip (chain, store_scr)
    return get_the_chain
    
def find_str_chain(chain):
    str_chain = []
    total_scr = []
    var_scr = []
    length = len(chain)
    for each in chain:
        total_scr.append(each[1])
    ##AVERAGE CHAIN SCORE
    avg_scr = (sum(total_scr))/length

    ##VARIANCE
    for each in chain:
        fir_scr = (each[1]-avg_scr)
        get_scr = math.pow(fir_scr, 2)
        var_scr.append(get_scr)

    var_chain = sum(var_scr)/length

    ##Standard Deviation of Chain score
    std_dev = math.sqrt(var_chain)

    ##Strength Criterion value
    str_crit = avg_scr+(2*std_dev)

    for each in chain:
        if each[1] > str_crit:
            str_chain.append(each)
    print 'Strength Criteion: ',str_crit,'\n'
    return str_chain

def sent_extract(chain, content, dir_nouns, rel_nouns):
    get_candidates = []
    sent_candidates = []
    get_keywords = []

    ##KEY WORDS
    for each in dir_nouns:
        if each[1] == 1:
            get_keywords.append(each)
        else:
            pass
    for each in rel_nouns:
        if each[1] == 0.5:
            get_keywords.append(each)
        else:
            pass

    ##GET CANDIDATE CHAIN KEYWORDS  
    for each in chain:
        for item in each[0]:
            get_candidates.append(item)
    
    get_keys = list(set(get_candidates))

    ##Candidate Sentences
    for each in get_keys:
        for sent in content:
            if each in sent[0]:
                sent_candidates.append(sent)
            else:
                pass

    final_sent = list(set(sent_candidates))

    ##ORDER SENTENCES
    marks = []
    store_sen = []
    for each in final_sent:
        temp_mark = 0
        for k in get_keywords:
            if k[0] in each[0]:
                temp_mark += 1
            else:
                temp_mark += 0
        marks.append(temp_mark)
      
    ##print len(final_sent) LENGTH OF FINAL SENTENCES
    get_final = zip(final_sent, marks)
    sorted_sen = [(tup[1], tup[0][2], tup[0][0]) for tup in get_final]
    final_sen = sorted(sorted_sen, reverse=True)

    return final_sen
        
def summary(keywords, title, content):
    
    ##Get Nouns from title
    tit_nouns = extractNouns(title)
    #Ranking Keyowrds relatedness with Title
    ##Synset with Title & Score
    key_nouns = []
    key_unouns = []
    key_rel_nouns = []
    pre_rel_nouns = []
    keyword = list(set(keywords))
    
    for k in keyword:
        if k in tit_nouns:
            key_nouns.append(1)
        else:
            key_nouns.append(0)
            key_unouns.append(k)
    key_nouns_scr = zip(keyword, key_nouns)
    #####
    for k in key_unouns:
        for t in tit_nouns:
            for s in wn.synsets(t):
                lemmas = s.lemmas()
                for l in lemmas:
                    if k in l.name():
                        pre_rel_nouns.append(k)
                    else:
                        pass
    
    get_rel_nouns = list(set(pre_rel_nouns))
    
    get_rel = []
    for n in key_unouns:
        if n in get_rel_nouns:
            get_rel.append(0.5)
        else:
            get_rel.append(0)

    get_rel_scr = zip(key_unouns, get_rel)
    
    ##print## key_nouns_scr [NOUNS WITH SCORE 1]
    ##print## get_rel_scr [NOUNS WITH SCORE 0.5]
    ##Get Sentence Information (str, #index_num, #length(# word in sen))
    modify_sent = get_sent(content)
    
    #Build Lexical Chains
    get_chains = find_chain(keywords, keyword)
    
    #Score Chains with Keyword Score, Frequency and Chain length
    chains_scr = score_chain(get_chains, key_nouns_scr, get_rel_scr)
    #Find Strong Chain
    strong_chain = find_str_chain(chains_scr)

    ##Extract summary sentences
    get_summary = sent_extract(strong_chain, modify_sent, key_nouns_scr, get_rel_scr)
    get_sum_len = len(get_summary)
    print '#############################'
    print title,'\n'
    print '########## SUMMARY ##########\n'
    
    if (get_sum_len>5):
        get_five = get_summary[:5]
        for each in get_five:
            print each[2],'\n'
    else:
        for each in get_summary:
            print each[2],'\n'
