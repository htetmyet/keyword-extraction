# -*- coding: utf-8 -*-
"""
Created on Tue Apr 04 16:25:18 2017

@author: Administrator
"""
import glob, random, os
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from decimal import*
from collections import Counter

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
    #Get paragraph without title
    para_list = raw_txt.splitlines()[1:] #in list
    
    get_cont = []
    for each in para_list:
        get_cont.append(each)
    
    return get_cont
    
    
def get_keywords(doc_name):
    file_path = 'traindata/'
    test_file = open(file_path+doc_name, 'r')
    rawtext = test_file.read()
    get_last = rawtext.splitlines()[-1]
    split_last = get_last.split(',')
    ##GET ALL KEYWORDS
    get_all_keywords = []
    for each in split_last:
        get_all_keywords.append(each)
    
    return get_all_keywords

def main():
    DIR = 'dataset/'
    data_list = []
    num_files = len(glob.glob1(DIR,"*.txt"))
    print 'Total ',num_files, ' files.'

    get_files = []
    cnt = 0
    num_of_turn = 10
    while (len(get_files) < num_of_turn):
        temp_file = random.choice([x for x in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, x))])
        if temp_file not in get_files:
            get_files.append(temp_file)
            cnt +=1
    
    for each_f in get_files:
        get_keys = get_keywords(each_f)
        ##print get_keys
        get_content = get_cont(each_f)
        
        first_data = []
        
        for index, each_sent in enumerate(get_content):
            index += 1
            get_data = each_sent, index
            first_data.append(get_data)
        
        new_norm = norm_pos(first_data)
        
        get_node = []
        for each in get_keys:
            for x in new_norm:
                if each in x[0]:
                    get_node.append(str(x[1]))
                else:
                    pass
        get_count = Counter(get_node)
        get_into_lst = list(get_count.items())
        doc_list = []
        for each in get_into_lst:
            new_list = each[0], each[1], each_f
            doc_list.append(new_list)
        data_list.append(doc_list) 
    for each in data_list:
        print each
    ##PLOT DATA    

    markers=[ur"$\u25A1$", ur"$\u25A0$", ur"$\u25B2$", ur"$\u25E9$"]
    colors= ["k", "crimson", "#112b77"]
    fig, ax = plt.subplots()
    for i, l  in enumerate(data_list):
        x,y,cat = zip(*l)
        ax.scatter(list(map(float, x)),y, s=64,c=colors[(i//4)%3],
                                      marker=markers[i%4], label=cat[0])

    ax.legend(bbox_to_anchor=(1.01,1), borderaxespad=0)
    plt.subplots_adjust(left=0.1,right=0.8)
    plt.show()
  
            
    
if __name__ == '__main__':
    main()
    