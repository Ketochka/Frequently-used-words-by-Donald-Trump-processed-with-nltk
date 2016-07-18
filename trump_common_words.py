
# coding: utf-8

# In[4]:

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

# Just getting a text with a link
def get_text(link):
    text = request.urlopen(link).read().decode('utf8')
    text = BeautifulSoup(text, "lxml").get_text()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub('\s+', ' ', text)
    return text

# Function for getting frequent words without frequency numbers
def freq_words(cutted_text, n):
    freqdist = nltk.FreqDist(word.lower() for word in word_tokenize(cutted_text))
    result = []
    for word, count in freqdist.most_common(n):
        if word.isalpha() and len(word)>5:
             result = result + [word]
    return result

# Function for tokenizing, stemming and fixing some stemming bugs
def tokenizing_and_stemming(url, start, finish):
    raw = get_text(url)
    raw = raw[start:finish]
    tokens_1 = word_tokenize(raw)
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    tokens_2 = []
    for each in tokens_1:
        if each=='death':
             tokens_2.append('living')
        elif each=='betrayed':
             tokens_2.append('appreciated') 
        elif each=='failed':
             tokens_2.append('succeeded')  
        elif each=='single':
             tokens_2.append('tandem')         
        elif each=='possibilities':
             tokens_2.append('possibility')
        elif each=='governments':
             tokens_2.append('government')
        elif each=='understand':
             tokens_2.append('understanding')
        elif each=='includ':
             tokens_2.append('include')
        elif each.startswith('immigr') and not each.endswith('ant'):
             tokens_2.append('immigration') 
        elif each.startswith('contempl'):
             tokens_2.append('contemplate') 
        elif each.startswith('refuge'):
             tokens_2.append('refugee')  
        elif each.startswith('speach') and not each.endswith('ch'):
             tokens_2.append('speachless')          
        elif (not each.endswith('aps') and not each.endswith('ous') and not each.endswith('eas')
             and each.endswith('s') or each.endswith('ies')):
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

# Getting 1000 most common words tagged with specific part of speach (first ordinary tagging) 
freq1000 = freq_words((raw_1+raw_2+raw_3), 1000)
tagged_thousand_freq = nltk.pos_tag(freq1000)

# Making patterns for second improved tagging
pats = [
        (r'.*(believe|declare|become|reject|discuss|protect|prevent)$', 'VB'),
        (r'.*(threaten|withdraw|imagine|continue|appoint)$', 'VB'),
        (r'.*(ment|ist|ee|tion|ant|thing|ound|cian)$', 'NN'),
        (r'.*(message|father|friend|refugee|muslim|speech)$', 'NN'), 
        (r'.*(server|service|screen|ambassador|regime)$', 'NN'),    
        (r'.*(killer|shooter|citizen|victim|september|november)$', 'NN'),    
        (r'.*(hillary|mendoza|orlando|donald|kasich|arabia)$', 'NNP'), 
        (r'.*(mexico|bernie|pennsylvania|america|afghanistan)$', 'NNP'),
        (r'.*ing$', 'VBG'),
        (r'.*ed$', 'VBD'),
        (r'.*(tonight|overseas|aram|where|sand)$', 'RB'),
        (r'.*(google|politico|facebook|twitter)$', 'AAA')

         ]
regexp_tagger = nltk.RegexpTagger(pats)

# Tagging only selected(which correspond to pattern) words
reg_tagged_thousand_freq = regexp_tagger.tag(freq1000)

# Changing tags after first tagging with tags after second tagging(for words which correspond to patterns)

for word, tag in sorted(tagged_thousand_freq):
    for word_, tag_ in sorted(reg_tagged_thousand_freq):
        if tag_!=None and word==word_:
            tagged_thousand_freq.remove((word,tag))
            tagged_thousand_freq.append((word_,tag_))
            
        

# Selecting specific parts of the words (e.g nouns, verbs..)
def pos_tagger(tagged_words, condition):
    parts = []
    for w,t in tagged_words:
              if t==condition and len(w)>=2:
                    parts.append(w)
    return parts 

# Defining condition lists
nouns = ['NN', 'NNP','NNS']
verbs = ['VB','VBD','VBG','VBN','VBP','VBZ']
adjectives = ['JJ','JJR','JJS']

# General function for appending something to some list
def appending(result_list, what_to_append):
    for each in what_to_append:
        result_list.append(each)

# Function for selecting specific parts of the words with their frequency + fixing some tagging bugs
def common_words_with_frequency(tagged_thousand_freq, raw, condition_list):
    words = []
    for i in range(0, len(condition_list)):
        appending(words, pos_tagger((tagged_thousand_freq), condition_list[i]))
    
    freqdist = nltk.FreqDist(word.lower() for word in word_tokenize(raw) if len(word)>=5)
    freq_with_counts = freqdist.most_common(1000)
    our_words = [item for item in freq_with_counts if item[0] in words if item[1]!=1]
    return our_words

# Getting most frequent nouns with some names with their frequencies
all_nouns = common_words_with_frequency(tagged_thousand_freq, (raw_1+raw_2+raw_3), nouns)
all_nouns = [item for item in all_nouns if item[1]>3]
all_verbs = common_words_with_frequency(tagged_thousand_freq, (raw_1+raw_2+raw_3), verbs)
all_verbs = [item for item in all_verbs if item[1]>=2]
all_adjectives = common_words_with_frequency(tagged_thousand_freq, (raw_1+raw_2+raw_3), adjectives)
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

# Function for generating wordcloud 
def generate_wordcloud(our_string, name):
    kiss_me = np.array(Image.open("kiss_me_.png"))
    wordcloud = WordCloud(font_path='C:\Windows\Boot\Fonts\malgun_boot.ttf',
                          stopwords=None,
                          background_color='black',
                          mask=kiss_me
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

