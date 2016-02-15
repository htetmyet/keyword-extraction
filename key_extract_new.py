from refo import finditer, Predicate, Plus
from collections import Counter

##from nltk.corpus import stopwords
import math
import nltk
import re
import copy
import glob

##cachedStopWords = stopwords.words("english")

def convert_to_string(text):
    get_index = text.index(':')
    get_length = len(text)
    get_string = text[0:get_index]
    return get_string

def remov_stopword(text):
    stopwords = open ('nothesmartstoplist.txt', 'r').read().splitlines()
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text

def get_title(text):
    pre_title = text.splitlines()[0]
    return pre_title

def get_first_sen(text):
    get_first = text.splitlines()[1]
    return get_first

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

def term_frequency(w_tf):
    tf_score = 1 + math.log10(w_tf)
    return tf_score

def count_total_corpus():
    tot_corpus =  len(glob.glob1("dataset","doc*.txt"))
    return tot_corpus

def count_nterm_doc(word):
    num_count = 0
    get_total = count_total_corpus()
    while (get_total>0):
        n_files = str(get_total)
        get_doc = open('dataset/doc'+n_files+'.txt', 'r')
        raw_doc = get_doc.read()
        if word in raw_doc:
           num_count += 1
        else:
            num_count += 0
        get_total -= 1
    return num_count

def inverse_df(tot_doc, num_of_x_doc):
    idf_score = math.log10(1+(tot_doc/num_of_x_doc))
    return idf_score

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

def get_bi_tfidf(bi_dict, bigrams):
    i=0
    for bigrams[i] in bigrams:
        for w in bigrams[i]:
            if w in bi_dict:
                print bi_dict[w]
                
def get_val_bipairs(bi_dict, bigrams):
    val_pairs = [(bi_dict[x]+bi_dict[y]) for x,y in bigrams]
    return val_pairs

def get_val_tripairs(tri_dict, trigrams):
    val_pairs = [(tri_dict[x]+tri_dict[y]+tri_dict[z]) for x,y,z in trigrams]
    return val_pairs

def get_val_fpairs(fgram_dict, fourgrams):
    val_pairs = [(fgram_dict[a]+fgram_dict[b]+fgram_dict[c]+fgram_dict[d]) for a,b,c,d in fourgrams]
    return val_pairs

def main():
    ##set_file_name = raw_input('Enter a file name: ')
    file_name = raw_input('Enter a file name: ')
    test_file = open(file_name, 'r')
    rawtext = test_file.read()
    
    #Extract title from text
    title = get_title(rawtext)
    first_sen = get_first_sen(rawtext)
    
    #Get paragraph without title
    para_list = rawtext.splitlines()[1:] #in list
    para_string = ''.join(para_list) #convert to string
    
    #Prettify paragraph
    prettify_txt = re.sub(r'[^\w.]', ' ', para_string)
    mod_txt = remov_stopword(prettify_txt)
    
    #Tokenizing & POS Tagging
    token_txt = nltk.sent_tokenize(mod_txt) #Line Segment
    
    num_sent = len(token_txt) #Number of sentences
    token_word = [nltk.word_tokenize(sent) for sent in token_txt]
    pos_tag = [nltk.pos_tag(sent) for sent in token_word]

    print title
    print "Sentence: ", num_sent
    print '\n'
    
    #Chunk and print NP
    get_nouns = [[Word(*x) for x in sent] for sent in pos_tag]

    #NNP Rules
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
    ###################################GET UNIGRAMS###################################
    print "UNIGRAM -->"
    for k, s in enumerate(get_nouns):
        for match in finditer(get_uni_gram, s):
            x, y = match.span() #the match spans x to y inside the sentence s
            ##print pos_tag[k][x:y]
            bag_of_NP += pos_tag[k][x:y]

    ###############      
    #Term Frequency for unigrams    
    print "\nUnigram Feature Matrices:"
    total__tfidf = 0
    uni_tfidf_values = ''
    str_uni_grams = ''
    total_docs = count_total_corpus()
    fdist = nltk.FreqDist(bag_of_NP)
    
    ##STORE UNIGRAMS
    unzipped_uni = zip(*bag_of_NP)
    str_unigrams = list(unzipped_uni[0])
    ##get_unigrams = zip(str_unigrams,str_unigrams[1:])[::2]
    ###############
    for word in fdist:
        fq_word = fdist[word]
        ##print '%s->%d' % (word, fq_word)
        get_tf = term_frequency(fq_word)

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

        ##GET EACH UNIGRAMS TFIDF
        uni_tfidf_scr = repr(tf_idf_scr)+' '
        uni_tfidf_values += uni_tfidf_scr
        str_uni_grams += get_this_string+','

    ##BUILD DICT FOR EACH TERMS
    get_uni_float = [float(x) for x in uni_tfidf_values.split()]
    get_uni_list = str_uni_grams.split(',')
    unigram_dict = dict(zip(get_uni_list, get_uni_float))
    ###########################  

    ##GET TFIDF FOR UNIGRAMS##
    ############
    uni_avg_tfidf = (sum(map(float,get_uni_float)))/(len(get_uni_float))
    ###########################
    get_zip_str = [''.join(item) for item in str_unigrams]
    ###Unigrams string with TFIDF###
    unigrams_list =  zip(get_zip_str, get_uni_float)
    ###########################
    
    print '===============***==============='
    print 'Total Unigrams: ', len(fdist)
    print 'Total tfidf', total__tfidf
    print 'Average TF.IDF: ', uni_avg_tfidf
    print '===============***==============='
    ###########################
    ##### TFIDF FEATURE MATRIX #####
    uni_feat_tfidf = []
    for x in unigrams_list:
        if float(x[1]) > uni_avg_tfidf:
            uni_feat_tfidf.append(1)
        else:
            uni_feat_tfidf.append(0)
    zip_tfidf_feat = zip(get_zip_str, get_uni_float, uni_feat_tfidf)
    ################################
    ##### First Sentence Feat #####
    uni_fir_sen = []
    for x in unigrams_list:
        get_res = chk_frs_sen(x[0], file_name)
        if get_res == 1:
            uni_fir_sen.append(1)
        else:
            uni_fir_sen.append(0)
    zip_fir_sen_feat = zip(get_zip_str, get_uni_float, uni_feat_tfidf, uni_fir_sen)
    ################################
    ##### Involve in Title #####
    uni_title_feat = []
    for x in unigrams_list:
        get_res = involve_in_title(x[0], title)
        if get_res == 1:
            uni_title_feat.append(1)
        else:
            uni_title_feat.append(0)
    zip_uni_feats = zip(get_zip_str, get_uni_float, uni_feat_tfidf, uni_fir_sen, uni_title_feat)
    print zip_uni_feats
    ################################
    print "\n\n"
    ###################################GET BIGRAMS###################################
    print "BIGRAM -->"
    for k, s in enumerate(get_nouns):
        for match in finditer(get_bi_gram, s):
            x, y = match.span()
            ##print pos_tag[k][x:y]
            bag_of_biNP += pos_tag[k][x:y]
            
    ##Term Frequency for bigrams##
    total__tfidf = 0
    bi_tfidf_values = ''
    str_bi_grams = ''
    ###############
    
    ##STORE BIGRAMS
    unzipped = zip(*bag_of_biNP)
    str_bigrams = list(unzipped[0])
    get_bigrams = zip(str_bigrams,str_bigrams[1:])[::2]
    
    ###############
    
    print "\nBigram Feature Matrices:"
    bi_dist = nltk.FreqDist(bag_of_biNP)
    for word in bi_dist:
        
        tq_word = bi_dist[word]
        ##print '%s-->%d' % (word, tq_word)
        get_tf = term_frequency(tq_word)
        
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
    ###########################
    ##print bigrams_list
    
    print '===============***==============='
    print 'Total Bigrams: ', len(get_bi_floats)
    print 'total tfidf: ', sum(map(float,get_bi_floats))
    print 'Average TF.IDF: ', real_avg_tfidf
    print '===============***==============='
    ##print len(bi_str2_float(bi_tfidf_values))
    ##print type(bag_of_biNP)
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
        get_res = chk_frs_sen(x[0], file_name)
        if get_res == 1:
            feat_fir_sen.append(1)
        else:
            feat_fir_sen.append(0)
            
    fir_sen_feat = zip (get_zip_str, get_bi_floats, feat_tfidf_matx, feat_fir_sen)
    ##print fir_sen_feat
    #################################
    #### INVOLVE IN TITLE FEATURE ###
    feat_invol_tit = []
    for x in fir_sen_feat:
        get_res = involve_in_title(x[0], title)
        if get_res == 1:
            feat_invol_tit.append(1)
        else:
            feat_invol_tit.append(0)
    invol_tit_feat = zip (get_zip_str, get_bi_floats, feat_tfidf_matx, feat_fir_sen, feat_invol_tit)
    print invol_tit_feat
    #################################
    print "\n\n"
    
    ###################################GET TRIGRAMS###################################
    print "TRIGRAM -->"
    for k, s in enumerate(get_nouns):
        for match in finditer(get_tri_gram, s):
            x, y = match.span()
            ##print pos_tag[k][x:y]
            bag_of_triNP += pos_tag[k][x:y]
            
    #Term Frequency for trigrams
    total__tfidf = 0
    tri_tfidf_values = ''
    str_tri_grams = ''
    ###############
    
    ##STORE TRIGRAMS
    unzipped_tri = zip(*bag_of_triNP)
    str_trigrams = list(unzipped_tri[0])
    get_trigrams = zip(str_trigrams,str_trigrams[1:],str_trigrams[2:])[::3]
    ###############
    
    print "\nTrigram Feature Matrices:"
    tri_dist = nltk.FreqDist(bag_of_triNP)
    for word in tri_dist:
        tr_fq = tri_dist[word]
        ##print '%s-->%d' % (word, tr_fq)
        get_tf = term_frequency(tr_fq)
    
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
    print '===============***==============='
    print 'Total Trigrams: ', len(get_tri_floats)
    print 'Total tfidf', sum(map(float,get_tri_floats))
    print 'Average TF.IDF: ', tri_avg_tfidf
    print '===============***==============='
    ##### TFIDF FEATURE MATRIX #####
    tri_tfidf_matx = []
    for x in trigrams_list:
        if float(x[1]) > tri_avg_tfidf:
            tri_tfidf_matx.append(1)
        else:
            tri_tfidf_matx.append(0)
            
    tri_tfidf_feat = zip(get_ziptri_str, get_tri_floats, tri_tfidf_matx)
    #################################
    #### FIRST SENTENCE FEATURE ####
    tri_fir_sen = []
    for x in tri_tfidf_feat:
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
    print tri_tit_feat
    #################################
    print "\n\n"

    ###################################GET 4-GRAMS###################################
    print "4th GRAM -->"
    for k, s in enumerate(get_nouns):
        for match in finditer(get_quard_gram, s):
            x,y = match.span()
            ##print pos_tag[k][x:y]
            bag_of_fourNP += pos_tag[k][x:y]

    #Term Frequency for 4-grams
    total__tfidf = 0
    four_tfidf_values = ''
    str_four_grams = ''
    ###############
    
    ##STORE 4-GRAMS
    unzipped_four = zip(*bag_of_fourNP)
    str_fourgrams = list(unzipped_four[0])
    get_fourgrams = zip(str_fourgrams,str_fourgrams[1:],str_fourgrams[2:],str_fourgrams[3:])[::4]
    ###############
    
    #Term Frequency for 4-grams
    total__tfidf = 0
    print "\n4-grams Feature Matrices:"
    f_dist = nltk.FreqDist(bag_of_fourNP)
    for word in f_dist:
        fr_fq = f_dist[word]
        ##print '%s-->%d' % (word, fr_fq)
        get_tf = term_frequency(fr_fq)

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
    print '===============***==============='
    print 'Total 4-grams: ', len(get_four_floats)
    print 'Total tfidf', sum(map(float,get_four_floats))
    print 'Average TF.IDF: ', four_avg_tfidf
    print '===============***==============='
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
    print four_tit_feat
    #################################
if __name__ == '__main__':
    main()

