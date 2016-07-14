
# coding: utf-8

# In[2]:

import nltk, re, pprint
from nltk import word_tokenize, sent_tokenize
from urllib import request
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import *
from nltk.stem.snowball import SnowballStemmer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from os import path
from PIL import Image
import numpy as np

url_1 = "http://www.politico.com/story/2016/06/transcript-donald-trump-national-security-speech-224273"
url_2 = "http://www.politico.com/story/2016/06/transcript-trump-speech-on-the-stakes-of-the-election-224654"
url_3 = "http://www.politico.com/story/2016/06/full-transcript-trump-job-plan-speech-224891"

def get_text(link):
    text = request.urlopen(link).read().decode('utf8')
    text = BeautifulSoup(text, "lxml").get_text()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub('\s+', ' ', text)
    return text

def freq_words(cutted_text, n):
    freqdist = nltk.FreqDist(word.lower() for word in word_tokenize(cutted_text))
    result = []
    for word, count in freqdist.most_common(n):
        if word.isalpha() and len(word)>5:
             result = result + [word]
    return result


def tokenizing_and_stemming(url, start, finish):
    raw = get_text(url)
    raw = raw[start:finish]
    tokens_1 = word_tokenize(raw)
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    tokens_2 = []
    exception_list = ['perhaps','disastrous', 'overseas']
    for each in tokens_1:
        if each=='death':
             tokens_2.append('living')
        if each=='possibilities':
             tokens_2.append('possibility')
        elif each=='governments':
             tokens_2.append('government')
        elif each.startswith('immigr') and not each.endswith('ant'):
             tokens_2.append('immigration')  
        elif each.startswith('refuge'):
             tokens_2.append('refugee')  
        elif each.startswith('speach') and not each.endswith('ch'):
             tokens_2.append('speachless')          
        elif each not in exception_list and each.endswith('s') or each.endswith('ies'):
              tokens_2.append(stemmer.stem(each))
        else:
             tokens_2.append(each)
    tokens = []            
    for each in tokens_2:
        if each.endswith('i'): 
            tokens.append(each[:-1]+'y')
        else:
            tokens.append(each) 

    text = nltk.Text(tokens)
    raw = ' '.join(tokens)
    return text, raw, tokens

# Getting our text, raw material and tokens
text_1, raw_1, tokens_1 = tokenizing_and_stemming(url_1,13745,31980)
text_2, raw_2, tokens_2 = tokenizing_and_stemming(url_2,13464,33190)
text_3, raw_3, tokens_3 = tokenizing_and_stemming(url_3,13506,28425)
# Getting 500 most common words tagged with specific part of speach 
tagged_five_hundred_freq = nltk.pos_tag(freq_words((raw_1+raw_2+raw_3), 500))

def pos_tagger(tagged_words, condition):
    parts = []
    for w,t in tagged_words:
              if t==condition and len(w)>=2:
                    parts.append(w)
    return parts 

nouns = ['NN', 'NNP','NNS']
verbs = ['VB','VBD','VBG','VBN','VBP','VBZ']
adjectives = ['JJ','JJR','JJS']

def appending(result_list, what_to_append):
    for each in what_to_append:
        result_list.append(each)
        
def common_words_with_frequency(tagged_five_hundred_freq, raw, condition_list):
    words = []
    for i in range(0, len(condition_list)):
        appending(words, pos_tagger((tagged_five_hundred_freq), condition_list[i]))
    
    nn_list = ['father','hillary','friend','mendoza','immigrant','immigration','orlando'
               ,'terrorist','increase','refugee','muslim','speech']    
    if 'NN' in condition_list:
        for i in range(0,len(nn_list)):
            if nn_list[i] not in words:
                words.append(nn_list[i])
                
    jj_list = ['terrorist','hillary','citizen','immigrant','friend','mendoza','immigration'
               ,'orlando','father','become','increase','reject','single','discuss','refugee']    
    if 'JJ' in condition_list:
        for i in range(0,len(jj_list)):
            if jj_list[i] in words:
                words.remove(jj_list[i])
        
    vb_list = ['regime','america','refugee','muslim','speech','terrorist']  
    vb_list_a = ['become','reject','discuss'] 
    if 'VB' in condition_list:
        for i in range(0,len(vb_list)):
            if vb_list[i] in words:
                words.remove(vb_list[i]) 
        for i in range(0,len(vb_list_a)):
            if vb_list_a[i] not in words:
                words.append(vb_list_a[i])         
           
    freqdist = nltk.FreqDist(word.lower() for word in word_tokenize(raw) if len(word)>=5)
    freq_with_counts = freqdist.most_common(1000)
    our_words = [item for item in freq_with_counts if item[0] in words if item[1]!=1]
    return our_words

# Getting most frequent nouns with some names with their frequencies
all_nouns = common_words_with_frequency(tagged_five_hundred_freq, (raw_1+raw_2+raw_3), nouns)
all_nouns = [item for item in all_nouns if item[1]>3]
all_verbs = common_words_with_frequency(tagged_five_hundred_freq, (raw_1+raw_2+raw_3), verbs)
all_verbs = [item for item in all_verbs if item[1]>=2]
all_adjectives = common_words_with_frequency(tagged_five_hundred_freq, (raw_1+raw_2+raw_3), adjectives)
all_adjectives = [item for item in all_adjectives if item[1]>=3]
freqdist = nltk.FreqDist(word.lower() for word in word_tokenize(raw_1+raw_2+raw_3) if len(word)>=5)
all_words = freqdist.most_common(1000)
all_words = [item for item in all_words if item[1]>=6]

def convert_to_string(our_tuples):
    words = []
    counts = []
    for w,t in our_tuples:
        words.append(w)
        counts.append(t)
    for i in range(0,len(our_tuples)):
        for j in range(1, counts[i]):
            words.append(words[i])
    our_string = ' '.join(words)
    return our_string      

all_nouns_string = convert_to_string(all_nouns)
all_verbs_string = convert_to_string(all_verbs)
all_adjectives_string = convert_to_string(all_adjectives)
all_words_string = convert_to_string(all_words)

def generate_wordcloud(our_string, name):
    kiss_me = np.array(Image.open("kiss_me_.png"))
    wordcloud = WordCloud(font_path='C:\Windows\Boot\Fonts\malgun_boot.ttf',
                          stopwords=None,
                          background_color='black',
                          mask=kiss_me
                          #width=1200,
                          #height=1000
                         ).generate(our_string)

    wordcloud.to_file('%s' %name)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.figure()
    plt.imshow(kiss_me, cmap=plt.cm.gray)
    plt.axis("off")
    #plt.show()
    
generate_wordcloud(all_nouns_string, "Nouns_and_names_used_by_Trump_in_June_2016.png")    
generate_wordcloud(all_verbs_string, "Verbs_used_by_Trump_in_June_2016.png")    
generate_wordcloud(all_adjectives_string, "Adjectives_used_by_Trump_in_June_2016.png")   
generate_wordcloud(all_words_string, "Most_frequently_used_words_by_Trump_in_June_2016.png")

