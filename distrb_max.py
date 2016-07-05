from refo import finditer, Predicate, Plus
from collections import Counter

import math, nltk, re, copy, glob, sys, numpy as np, os

def get_val_bipairs(bi_dict, bigrams):
    val_pairs = [(bi_dict[x]+bi_dict[y]) for x,y in bigrams]
    return val_pairs

def get_val_tripairs(tri_dict, trigrams):
    val_pairs = [(tri_dict[x]+tri_dict[y]+tri_dict[z]) for x,y,z in trigrams]
    return val_pairs

def get_val_fpairs(fgram_dict, fourgrams):
    val_pairs = [(fgram_dict[a]+fgram_dict[b]+fgram_dict[c]+fgram_dict[d]) for a,b,c,d in fourgrams]
    return val_pairs

def term_frequency(w_tf, max_scr):
    tf_score = 0.5 + (0.5*(w_tf/max_scr))
    return tf_score

def count_total_corpus():
    tot_corpus =  len(glob.glob1("traindata","doc*.txt"))
    return tot_corpus

def count_nterm_doc(word):
    num_count = 0
    get_total = count_total_corpus()
    while (get_total>0):
        n_files = str(get_total)
        get_doc = open('traindata/doc'+n_files+'.txt', 'r')
        raw_doc = get_doc.read()
        if word in raw_doc:
           num_count += 1
        else:
            num_count += 0
        get_total -= 1
    return num_count

def inverse_df(tot_doc, num_of_x_doc):
    try:
        idf_score = math.log10(1+(tot_doc/num_of_x_doc))
    except ZeroDivisionError:
        idf_score = 0
    return idf_score

def convert_to_string(text):
    get_index = text.index(':')
    get_length = len(text)
    get_string = text[0:get_index]
    return get_string

def remov_stopword(text):
    stopwords = open ('nothesmartstoplist.txt', 'r').read().splitlines()
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text

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

def chk_keyword (word, n_grams):
    result = 0
    if word in n_grams:
        result = 1
    else:
        result = 0
    return result
        
def chk_frs_sen(word, file_name):##1 or 0 (binary)
    test_file = open(file_name, 'r')
    rawtext = test_file.read()
    first_sen = get_first_sen(rawtext)
    result_this = 0
    if word in first_sen:
        result_this = 1
    else:
        result_this = 0
    return result_this

def involve_in_title(word, get_title):
    result_this = 0
    if word in get_title:
        result_this = 1
    else:
        result_this = 0
    return result_this

def cal_bayes(n_grams):## DISTRIBUTING LIKELIHOOD
    prior_k = 0
    prior_nk = 0
    prior_tf = 0
    prior_ntf = 0
    prior_tit = 0
    prior_ntit = 0
    prior_fs = 0
    prior_nfs = 0

    total_key = 0
    total_nkey = 0
    tfk = 0
    tfnk = 0
    ntfk = 0
    ntfnk = 0
    fsk = 0
    fsnk = 0
    nfsk = 0
    nfsnk = 0
    titk = 0
    titnk = 0
    ntitk = 0
    ntitnk = 0
    
    for item in n_grams:
        if (item[5]==1):
            total_key += 1
        elif (item[5]==0):
            total_nkey += 1

    key_float = float(total_key)
    nkey_float = float(total_nkey)

    for item in n_grams:##Likelihood for P(TFIDF|K)
        if (item[2] ==1 and item[5] == 1):
            tfk += 1
    tfk_float = float(tfk)
    likeli_tfk = float(tfk_float/key_float)

    for item in n_grams:##Likelihood for P(TFIDF|NK)
        if (item[2] == 1 and item[5] == 0):
            tfnk += 1
    tfnk_float = float(tfnk)
    likeli_tfnk = float(tfnk_float/nkey_float)

    for item in n_grams:##Likelihood for P(NTFIDF|K)
        if (item[2] == 0 and item[5] == 1):
            ntfk += 1
    ntfk_float = float(ntfk)
    likeli_ntfk = float(ntfk_float/key_float)

    for item in n_grams:##Likelihood for P(NTFIDF|NK)
        if (item[2] == 0 and item[5] == 0 ):
            ntfnk += 1
    ntfnk_float = float(ntfnk)
    likeli_ntfnk = float(ntfnk_float/nkey_float)

    ##SEEK PRIOR P(TFIDF == 1) AND P(TFIDF == 0)
    prior_tf = float(likeli_tfk + likeli_tfnk)
    prior_ntf = float(likeli_ntfk + likeli_ntfnk)
    #########################################

    for item in n_grams:##Likelihood for P(FIRSEN|K)
        if (item[3] == 1 and item[5] == 1):
            fsk += 1
    fsk_float = float(fsk)
    likeli_fsk = float(fsk_float/key_float)

    for item in n_grams:##Likelihood for P(FIRSEN|NK)
        if (item[3] == 1 and item[5] == 0):
            fsnk += 1
    fsnk_float = float(fsnk)
    likeli_fsnk = float(fsnk_float/nkey_float)

    for item in n_grams:##Likelihood for P(NFS|K)
        if (item[3] == 0 and item[5] == 1):
            nfsk += 1
    nfsk_float = float(nfsk)
    likeli_nfsk = float(nfsk_float/key_float)

    for item in n_grams:##Likelihood for P(NFS|NK)
        if (item[3] == 0 and item[5] == 0):
            nfsnk += 1
    nfsnk_float = float(nfsnk)
    likeli_nfsnk = float(nfsnk_float/nkey_float)

    ##SEEK PRIOR P(FIRSEN == 1) and P(FIRSEN == 0)
    prior_fs = float(likeli_fsk + likeli_fsnk)
    prior_nfs = float(likeli_nfsk + likeli_nfsnk)
    #########################################

    for item in n_grams:##Likelihood for P(TIT|K)
        if (item[4] == 1 and item[5] == 1):
            titk += 1
    titk_float = float(titk)
    likeli_titk = float(titk_float/key_float)

    for item in n_grams:##Likelihood for P(TIT|NK)
        if (item[4] == 1 and item[5] == 0):
            titnk += 1
    titnk_float = float(titnk)
    likeli_titnk = float(titnk_float/nkey_float)

    for item in n_grams:##Likelihood for P(NTIT|K)
        if(item[4]==0 and item[5]==1):
            ntitk += 1
    ntitk_float = float(ntitk)
    likeli_ntitk = float(ntitk_float/key_float)

    for item in n_grams:##Likelihood for P(NTIT|NK)
        if(item[4]==0 and item[5]==0):
            ntitnk += 1
    ntitnk_float = float(ntitnk)
    likeli_ntitnk = float(ntitnk_float/nkey_float)

    ##SEEK PRIOR P(TIT ==1) and P(TIT == 0)
    prior_tit = float(likeli_titk + likeli_titnk)
    prior_ntit = float(likeli_ntitk + likeli_ntitnk)
    ############################################

    prior_k = float(likeli_tfk + likeli_ntfk + likeli_fsk + likeli_nfsk + likeli_titk + likeli_ntitk)
    prior_nk = float(likeli_tfnk + likeli_ntfnk + likeli_fsnk + likeli_nfsnk + likeli_titnk + likeli_ntitnk)

    priors_feats =  prior_k, prior_nk, prior_tf, prior_ntf, prior_fs, prior_nfs, prior_tit, prior_ntit
    likehoods = likeli_tfk, likeli_tfnk, likeli_ntfk, likeli_ntfnk, likeli_fsk, likeli_fsnk, likeli_nfsk, likeli_nfsnk, likeli_titk, likeli_titnk, likeli_ntitk, likeli_ntitnk

    prior_ke = priors_feats[0]
    prior_nke = priors_feats[1]
    prior_tfe = priors_feats[2]
    prior_ntfe = priors_feats[3]
    prior_fse = priors_feats[4]
    prior_nfse = priors_feats[5]
    prior_tite = priors_feats[6]
    prior_ntite = priors_feats[7]
    likeli_tfke = likehoods[0]
    likeli_tfnke = likehoods[1]
    likeli_ntfke = likehoods[2]
    likeli_ntfnke = likehoods[3]
    likeli_fske = likehoods[4]
    likeli_fsnke = likehoods[5]
    likeli_nfske = likehoods[6]
    likeli_nfsnke = likehoods[7]
    likeli_titke = likehoods[8]
    likeli_titnke = likehoods[9]
    likeli_ntitke = likehoods[10]
    likeli_ntitnke = likehoods[11]

    try:
        pospro_ktf = float((likeli_tfke*prior_ke)/prior_tfe)
    except ZeroDivisionError:
        pospro_ktf = 0.1
    try:
        pospro_kntf = float((likeli_ntfke*prior_ke)/prior_ntfe)
    except ZeroDivisionError:
        pospro_kntf = 0.1
    try:
        pospro_nktf = float((likeli_tfnke*prior_nke)/prior_tfe)
    except ZeroDivisionError:
        pospro_nktf = 0.1
    try:
        pospro_nkntf = float((likeli_ntfnke*prior_nke)/prior_ntfe)
    except ZeroDivisionError:
        pospro_nkntf = 0.1
    try:
        pospro_kfs = float((likeli_fske*prior_ke)/prior_fse)
    except ZeroDivisionError:
        pospro_kfs = 0.1
    try:
        pospro_knfs = float((likeli_nfske*prior_ke)/prior_nfse)
    except ZeroDivisionError:
        pospro_knfs = 0.1
    try:
        pospro_nkfs = float((likeli_fsnke*prior_nke)/prior_fse)
    except ZeroDivisionError:
        pospro_nkfs = 0.1
    try:
        pospro_nknfs = float((likeli_nfsnke*prior_nke)/prior_nfse)
    except ZeroDivisionError:
        pospro_nknfs = 0.1
    try:
        pospro_ktit = float((likeli_titke*prior_ke)/prior_tite)
    except ZeroDivisionError:
        pospro_ktit = 0.1
    try:
        pospro_kntit = float((likeli_ntitke*prior_ke)/prior_ntite)
    except ZeroDivisionError:
        pospro_kntit = 0.1
    try:
        pospro_nktit = float((likeli_titnke*prior_nk)/prior_tite)
    except ZeroDivisionError:
        pospro_nktit = 0.1
    try:
        pospro_nkntit = float((likeli_ntitnke*prior_nke)/prior_ntite)
    except ZeroDivisionError:
        pospro_nkntit = 0.1
    
    val_bayes = pospro_ktf,pospro_kntf,pospro_nktf,pospro_nkntf,pospro_kfs,pospro_knfs,pospro_nkfs,pospro_nknfs,pospro_ktit,pospro_kntit,pospro_nktit,pospro_nkntit
    return val_bayes

##def dist_initial(n_grams, total_num):

def dist_tfidf (tuple_vals):
    this_kk = str(tuple_vals[0])
    this_knk = str(tuple_vals[1])
    this_nkk = str(tuple_vals[2])
    this_nknk = str(tuple_vals[3])

    tfidf_matx = np.matrix('"'+this_kk+' '+this_nkk+'; '+this_knk+' '+this_nknk+'"')
    return tfidf_matx

def dist_firsen (tuple_vals):
    this_kk = str(tuple_vals[4])
    this_knk = str(tuple_vals[5])
    this_nkk = str(tuple_vals[6])
    this_nknk = str(tuple_vals[7])
    
    firsen_matx = np.matrix('"'+this_kk+' '+this_nkk+'; '+this_knk+' '+this_nknk+'"')
    return firsen_matx

def dist_title(tuple_vals):
    this_kk = str(tuple_vals[8])
    this_knk = str(tuple_vals[9])
    this_nkk = str(tuple_vals[10])
    this_nknk = str(tuple_vals[11])
        
    title_matx = np.matrix('"'+this_kk+' '+this_nkk+'; '+this_knk+' '+this_nknk+'"')
    return title_matx

def matrix_txt(filename, matrix):
    file_name = 'matrices/'+filename
    if os.path.exists(file_name):
        os.remove(file_name)
        print 'Existing file will be replaced with a new file...'
        with open(file_name, 'w') as f:
            for line in matrix:
                np.savetxt(f, line, fmt = '%.2f')
            print 'File has been replaced.'
    else:
        print 'Distributing new file...'
        with open(file_name, 'w') as f:
            for line in matrix:
                np.savetxt(f, line, fmt = '%.2f')
            print 'File has been successfully distributed.'
    
def main():
    get_total = count_total_corpus()
    count = 0
    f_name = str(count+1)
    
    uni_collection = []
    bi_collection = []
    tri_collection = []
    four_collection = []
    
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

        key_unigram = ''
        key_bigram = ''
        key_trigram = ''
        key_fourgram = ''
        key_unknown = ''
        
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
            
        ### GET IN LIST ###
        key_unis = key_unigram.split(',')
        key_bis = key_bigram.split(',')
        key_tris = key_trigram.split(',')
        key_fours = key_fourgram.split(',')
        key_uns = key_unknown.split(',')
        ##print key_unis, key_bis, key_tris, key_fours, key_uns
            
        get_content = raw_doc.splitlines()[1:] #List form
        after_last_sen = get_content[:-1]
        content_str = ''.join(after_last_sen) #content in String format
        
        prettify_txt = re.sub(r'[^\w.]',' ', content_str)
        ##mod_txt = remov_stopword(prettify_txt)
        token_txt = nltk.sent_tokenize(prettify_txt)
        ##Number of Sentence: len(token_txt)##
        token_word = [nltk.word_tokenize(sent) for sent in token_txt]
        pos_tag = [nltk.pos_tag(sent) for sent in token_word]
<<<<<<< Updated upstream

=======
        ##print key_unigram, key_bigram, key_trigram, key_fourgram, key_unknown
>>>>>>> Stashed changes
        
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

        ##### GETTING EACH WORD TFIDF #####
        uni_tfidf_values = ''
        str_uni_grams = ''
        total_docs = count_total_corpus()
        fdist = nltk.FreqDist(bag_of_NP)

        unzip_unigram = zip(*bag_of_NP)
        str_unigrams = list(unzip_unigram[0])
        
        ##UNI MAXIMUM TermScore##
        scores = []
        for word in fdist:
            score = fdist[word]
            scores.append(score)
        max_uni = max(scores)
        ######################
        
        for word in fdist:
            fq_word = fdist[word]
            get_tf = term_frequency(fq_word, max_uni)

            to_string = ':'.join(word)
            get_this_string = convert_to_string(to_string)

            num_of_doc_word = count_nterm_doc(get_this_string)
            idf_score = inverse_df(total_docs, num_of_doc_word)

            tf_idf_scr = get_tf * idf_score
            total__tfidf += tf_idf_scr

            uni_tfidf_scr = repr(tf_idf_scr)+' '
            uni_tfidf_values += uni_tfidf_scr
            str_uni_grams += get_this_string+','

        get_uni_float = [float(x) for x in uni_tfidf_values.split()]
        get_uni_list = str_uni_grams.split(',')
        unigram_dict = dict(zip(get_uni_list, get_uni_float))
<<<<<<< Updated upstream
        
=======
        ##print unigram_dict
>>>>>>> Stashed changes
        ##### GET TFIDF FOR UNIGRAMS & AVERAGE TFIDF VALUES #####
        uni_avg_tfidf = (sum(map(float, get_uni_float)))/(len(get_uni_float))
        get_zip_str = [''.join(item) for item in str_unigrams]
        unigrams_list = zip(get_zip_str, get_uni_float)

        ##### TFIDF FEATURE MATRIX #####
        uni_feat_tfidf = []
        for x in unigrams_list:
            if float(x[1]) > uni_avg_tfidf:
                uni_feat_tfidf.append(1)
            else:
                uni_feat_tfidf.append(0)
        zip_tfidf_feat = zip(get_zip_str, get_uni_float, uni_feat_tfidf)
        ###############################
        ##### First Sentence Feat #####
        uni_fir_sen = []
        for x in unigrams_list:
            file_name = 'traindata/doc'+f_name+'.txt'
            get_res = chk_frs_sen(x[0], file_name)
            if get_res == 1:
                uni_fir_sen.append(1)
            else:
                uni_fir_sen.append(0)
        zip_fir_sen_feat = zip(get_zip_str, get_uni_float, uni_feat_tfidf, uni_fir_sen)
        ############################
        ##### Involve in Title #####
        uni_title_feat = []
        for x in unigrams_list:
            get_res = involve_in_title(x[0], title)
            if get_res == 1:
                uni_title_feat.append(1)
            else:
                uni_title_feat.append(0)
        zip_uni_feats = zip(get_zip_str, get_uni_float, uni_feat_tfidf, uni_fir_sen, uni_title_feat)
        ############################
        ##### KEYWORD OR NOT #####
        key_uni_matx = []
        for x in unigrams_list:
            get_res = chk_keyword(x[0],key_unis)
            if get_res == 1:
                key_uni_matx.append(1)
            else:
                key_uni_matx.append(0)
        zip_uni_all_feat = zip(get_zip_str, get_uni_float, uni_feat_tfidf, uni_fir_sen, uni_title_feat, key_uni_matx)
        #########################################################
        
        ##### GETTING BIGRAMS #####
        ##Term Frequency for bigrams##
        total__tfidf = 0
        bi_tfidf_values = ''
        str_bi_grams = ''
        
        unzip_bigram = zip(*bag_of_biNP)
        str_bigrams = list(unzip_bigram[0])
        get_bigrams = zip(str_bigrams, str_bigrams[1:])[::2]
        bi_dist = nltk.FreqDist(bag_of_biNP)

        ##BI MAXIMUM TermScore##
        bi_scores = []
        for word in bi_dist:
            score = bi_dist[word]
            bi_scores.append(score)
        max_bi = max(bi_scores)
        ######################
    
        for word in bi_dist:
            tq_word = bi_dist[word]
            get_tf = term_frequency(tq_word, max_bi)
        
            ### FEATURES ###
            ##Tuple to String##
            to_string = ':'.join(word)
            get_this_string = convert_to_string(to_string)
        
            ##DF Score
            num_of_doc_word = count_nterm_doc(get_this_string)
        
            ##TF.IDF Score
            idf_score = inverse_df(total_docs, num_of_doc_word)
            tf_idf_scr = get_tf*idf_score
            total__tfidf += tf_idf_scr

            ##GET EACH BIGRAMS TFIDF
            get_tfidf_scr = repr(tf_idf_scr)+' '
            bi_tfidf_values += get_tfidf_scr
            str_bi_grams += get_this_string+','
            
        ##BUILD DICT FOR EACH TERMS
        get_float = [float(x) for x in bi_tfidf_values.split()]
        get_bi_list = str_bi_grams.split(',')
        bigram_dict = dict(zip(get_bi_list, get_float))
        ###########################
    
        ##GET TFIDF FOR BIGRAMS##
        get_bi_floats = get_val_bipairs(bigram_dict, get_bigrams)
        get_zip = dict(zip(get_bigrams, get_bi_floats))
        ############
        real_avg_tfidf = (sum(map(float,get_bi_floats)))/(len(get_bi_floats))
        ###########################
        get_zip_str = [' '.join(item) for item in get_bigrams]
        ###Bigrams string with TFIDF###
        bigrams_list =  zip(get_zip_str, get_bi_floats)

        ##### TFIDF FEATURE MATRIX #####
        feat_tfidf_matx = []
        for x in bigrams_list:
            if float(x[1]) > real_avg_tfidf:
                feat_tfidf_matx.append(1)
            else:
                feat_tfidf_matx.append(0)
            
        tfidf_feat = zip(get_zip_str, get_bi_floats, feat_tfidf_matx)
        #################################
        #### FIRST SENTENCE FEATURE ####
        feat_fir_sen = []
        for x in tfidf_feat:
            file_name = 'traindata/doc'+f_name+'.txt'
            get_res = chk_frs_sen(x[0], file_name)
            if get_res == 1:
                feat_fir_sen.append(1)
            else:
                feat_fir_sen.append(0)
            
        fir_sen_feat = zip (get_zip_str, get_bi_floats, feat_tfidf_matx, feat_fir_sen)

        #### INVOLVE IN TITLE FEATURE ###
        feat_invol_tit = []
        for x in fir_sen_feat:
            get_res = involve_in_title(x[0], title)
            if get_res == 1:
                feat_invol_tit.append(1)
            else:
                feat_invol_tit.append(0)
        invol_tit_feat = zip (get_zip_str, get_bi_floats, feat_tfidf_matx, feat_fir_sen, feat_invol_tit)
        ##### KEYWORD OR NOT #####
        key_bi_matx = []
        for x in bigrams_list:
            get_res = chk_keyword(x[0],key_bis)
            if get_res == 1:
                key_bi_matx.append(1)
            else:
                key_bi_matx.append(0)
        zip_bi_all_feat = zip(get_zip_str, get_bi_floats, feat_tfidf_matx, feat_fir_sen, feat_invol_tit, key_bi_matx)
        #####################################
        ##### GETTING TRIGRAMS #####
        #Term Frequency for trigrams
        total__tfidf = 0
        tri_tfidf_values = ''
        str_tri_grams = ''
        
        unzip_trigram = zip(*bag_of_triNP)
        str_trigrams = list(unzip_trigram[0])
        get_trigrams = zip(str_trigrams, str_trigrams[1:], str_trigrams[2:])[::3]
        tri_dist = nltk.FreqDist(bag_of_triNP)

        ##TRI MAXIMUM TermScore##
        tri_scores = []
        for word in tri_dist:
            score = tri_dist[word]
            tri_scores.append(score)
        max_tri = max(tri_scores)
        ######################
    
        for word in tri_dist:
            tr_fq = tri_dist[word]
            get_tf = term_frequency(tr_fq, max_tri)
    
            ### FEATURES ###
            ##Tuple to String##
            to_string = ':'.join(word)
            get_this_string = convert_to_string(to_string)
            ##DF Score
            num_of_doc_word = count_nterm_doc(get_this_string)
            ##
            ##TF.IDF Score
            idf_score = inverse_df(total_docs, num_of_doc_word)
            tf_idf_scr = get_tf * idf_score
            total__tfidf += tf_idf_scr

            ##GET EACH TRIGRAMS TFIDF
            get_tfidf_scr = repr(tf_idf_scr)+' '
            tri_tfidf_values += get_tfidf_scr
            str_tri_grams += get_this_string+','
            
        ##BUILD DICT FOR EACH TERMS
        get_tri_float = [float(x) for x in tri_tfidf_values.split()]
        get_tri_list = str_tri_grams.split(',')
        trigram_dict = dict(zip(get_tri_list, get_tri_float))
        ###########################
    
        ##GET TFIDF FOR TRIGRAMS##
        get_tri_floats = get_val_tripairs(trigram_dict, get_trigrams)
        get_tri_zip = dict(zip(get_trigrams, get_tri_floats))
        ############
        tri_avg_tfidf = (sum(map(float,get_tri_floats)))/(len(get_tri_floats))
        ###########################
        get_ziptri_str = [' '.join(item) for item in get_trigrams]
        ###Bigrams string with TFIDF###
        trigrams_list =  zip(get_ziptri_str, get_tri_floats)
        ###########################
        ##### TFIDF FEATURE MATRIX #####
        tri_tfidf_matx = []
        for x in trigrams_list:
            if float(x[1]) > tri_avg_tfidf:
                tri_tfidf_matx.append(1)
            else:
                tri_tfidf_matx.append(0)
            
        tri_tfidf_feat = zip(get_ziptri_str, get_tri_floats, tri_tfidf_matx)
        ################################
        #### FIRST SENTENCE FEATURE ####
        tri_fir_sen = []
        for x in tri_tfidf_feat:
            file_name = 'traindata/doc'+f_name+'.txt'
            get_res = chk_frs_sen(x[0], file_name)
            if get_res == 1:
                tri_fir_sen.append(1)
            else:
                tri_fir_sen.append(0)
            
        tri_sen_feat = zip (get_ziptri_str, get_tri_floats, tri_tfidf_matx, tri_fir_sen)
        #################################
        #### INVOLVE IN TITLE FEATURE ###
        tri_invol_tit = []
        for x in tri_sen_feat:
            get_res = involve_in_title(x[0], title)
            if get_res == 1:
                tri_invol_tit.append(1)
            else:
                tri_invol_tit.append(0)
        tri_tit_feat = zip (get_ziptri_str, get_tri_floats, tri_tfidf_matx, tri_fir_sen, tri_invol_tit)
        ##################################################
        ##### KEYWORD OR NOT #####
        key_tri_matx = []
        for x in trigrams_list:
            get_res = chk_keyword(x[0],key_tris)
            if get_res == 1:
                key_tri_matx.append(1)
            else:
                key_tri_matx.append(0)
        zip_tri_all_feat = zip(get_ziptri_str, get_tri_float, tri_tfidf_matx, tri_fir_sen, tri_invol_tit, key_tri_matx)
        #########################################################
        ##### GETTING 4-GRAMS #####
        #Term Frequency for 4-grams
        if (len(bag_of_fourNP)>0):
            
            total__tfidf = 0
            four_tfidf_values = ''
            str_four_grams = ''
            ###############
            unzip_fourgram = zip(*bag_of_fourNP)
            str_fourgrams = list(unzip_fourgram[0])
            get_fourgrams = zip(str_fourgrams, str_fourgrams[1:], str_fourgrams[2:], str_fourgrams[3:])[::4]
            ############################
            f_dist = nltk.FreqDist(bag_of_fourNP)
            ##4 MAXIMUM TermScore##
            four_scores = []
            for word in f_dist:
                score = f_dist[word]
                four_scores.append(score)
            max_four = max(four_scores)
            ######################
            
            for word in f_dist:
                fr_fq = f_dist[word]
                get_tf = term_frequency(fr_fq, max_four)

                ### FEATURES ###
                ##Tuple to String##
                to_string = ':'.join(word)
                get_this_string = convert_to_string(to_string)
                ##DF Score
                num_of_doc_word = count_nterm_doc(get_this_string)
                ##TF.IDF Score
                idf_score = inverse_df(total_docs, num_of_doc_word)
                tf_idf_scr = get_tf * idf_score
                total__tfidf += tf_idf_scr

                ##GET EACH FOURGRAMS TFIDF
                get_tfidf_scr = repr(tf_idf_scr)+' '
                four_tfidf_values += get_tfidf_scr
                str_four_grams += get_this_string+','

            ##BUILD DICT FOR EACH TERMS
            get_four_float = [float(x) for x in four_tfidf_values.split()]
            get_four_list = str_four_grams.split(',')
            fourgram_dict = dict(zip(get_four_list, get_four_float))
            ###########################

            ##GET TFIDF FOR 4-GRAMS##
            get_four_floats = get_val_fpairs(fourgram_dict, get_fourgrams)
            get_four_zip = dict(zip(get_fourgrams, get_four_floats))
            ############
            four_avg_tfidf = (sum(map(float,get_four_floats)))/(len(get_four_floats))
            ###########################
            get_zipfour_str = [' '.join(item) for item in get_fourgrams]
            ###Bigrams string with TFIDF###
            fourgrams_list =  zip(get_zipfour_str, get_four_floats)
            ###########################
            ##### TFIDF FEATURE MATRIX #####
            four_tfidf_matx = []
            for x in fourgrams_list:
                if float(x[1]) > four_avg_tfidf:
                   four_tfidf_matx.append(1)
                else:
                    four_tfidf_matx.append(0)
            
            four_tfidf_feat = zip(get_zipfour_str, get_four_floats, four_tfidf_matx)
            #################################
            #### FIRST SENTENCE FEATURE ####
            four_fir_sen = []
            for x in four_tfidf_feat:
                file_name = 'traindata/doc'+f_name+'.txt'
                get_res = chk_frs_sen(x[0], file_name)
                if get_res == 1:
                    four_fir_sen.append(1)
                else:
                    four_fir_sen.append(0)
            
            four_sen_feat = zip (get_zipfour_str, get_four_floats, four_tfidf_matx, four_fir_sen)
            #################################
            #### INVOLVE IN TITLE FEATURE ###
            four_invol_tit = []
            for x in tri_sen_feat:
                get_res = involve_in_title(x[0], title)
                if get_res == 1:
                    four_invol_tit.append(1)
                else:
                    four_invol_tit.append(0)
            four_tit_feat = zip (get_zipfour_str, get_four_floats, four_tfidf_matx, four_fir_sen, four_invol_tit)
            ##### KEYWORD OR NOT #####
            key_four_matx = []
            for x in fourgrams_list:
                get_res = chk_keyword(x[0],key_fours)
                if get_res == 1:
                    key_four_matx.append(1)
                else:
                    key_four_matx.append(0)
            zip_four_all_feat = zip(get_zipfour_str, get_four_floats, four_tfidf_matx, four_fir_sen, four_invol_tit, key_four_matx)
            #########################################################
        else:
            print 'Pass4-gram'
            zip_four_all_feat = ''
        
        uni_collection +=  zip_uni_all_feat
        bi_collection += zip_bi_all_feat
        tri_collection += zip_tri_all_feat
        four_collection += zip_four_all_feat
        
        total_unigram = len(uni_collection) ##UNIGRAM
        total_bigram = len(bi_collection) ##BIGRAM
        total_trigram = len(tri_collection) ##TRIGRAM
        total_fourgram = len(four_collection) ##FOURGRAM
        #######################
        print "Document "+n_files+" has been processed."
        count += 1

    ############################################
    get_uni_vals = cal_bayes(uni_collection)
    get_bi_vals = cal_bayes(bi_collection)
    get_tri_vals = cal_bayes(tri_collection)
    get_four_vals = cal_bayes(four_collection)
    ##### GET TFIDF DISTRIBUTIONS #####
    print '########## TFIDF DISTRIBUTIONS FOR N-GRAMS ##########'
    print dist_tfidf(get_uni_vals)
    print dist_tfidf(get_bi_vals)
    print dist_tfidf(get_tri_vals)
    print dist_tfidf(get_four_vals)
    ###################################
    ##### GET FIRST SENTENCE DISTRIBUTIONS #####
    print '########## FIRST SEN. DISTRIBUTIONS FOR N-GRAMS ##########'
    print dist_firsen(get_uni_vals)
    print dist_firsen(get_bi_vals)
    print dist_firsen(get_tri_vals)
    print dist_firsen(get_four_vals)
    ############################################
    ##### GET TITLE DISTRIBUTIONS #####
    print '########## TITLE DISTRIBUTIONS FOR N-GRAMS ##########'
    print dist_title(get_uni_vals)
    print dist_title(get_bi_vals)
    print dist_title(get_tri_vals)
    print dist_title(get_four_vals)
    ###################################

    ##### PRODUCE TEXT #####
    print '########## STORE INTO TEXT ##########'
    matrix_txt('uni_tf.txt',dist_tfidf(get_uni_vals))
    matrix_txt('uni_fs.txt',dist_firsen(get_uni_vals))
    matrix_txt('uni_tit.txt',dist_title(get_uni_vals))

    matrix_txt('bi_tf.txt',dist_tfidf(get_bi_vals))
    matrix_txt('bi_fs.txt',dist_firsen(get_bi_vals))
    matrix_txt('bi_tit.txt',dist_title(get_bi_vals))

    matrix_txt('tri_tf.txt',dist_tfidf(get_tri_vals))
    matrix_txt('tri_fs.txt',dist_firsen(get_tri_vals))
    matrix_txt('tri_tit.txt',dist_title(get_tri_vals))

    matrix_txt('four_tf.txt',dist_tfidf(get_four_vals))
    matrix_txt('four_fs.txt',dist_firsen(get_four_vals))
    matrix_txt('four_tit.txt',dist_title(get_four_vals))
    
if __name__ == '__main__':
    main()
