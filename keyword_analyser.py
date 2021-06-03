import re
import nltk
import pandas as pd
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns

# Create a list of stop words from nltk
stop_words = set(stopwords.words("english"))
other_stop_words = ['pay','current','month','much','per','need','get']
stop_words = stop_words.union(other_stop_words)

# View most frequently occuring keywords
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in      
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                       reverse=True)
    return words_freq[:n]

# Most frequently occuring bigrams
def get_top_n2_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(2,2),  
            max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]

# Most frequently occuring Tri-grams
def get_top_n3_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(3,3), 
           max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]

def analyse_words(df):

    corpus = []
    df['word_count'] = df['body'].apply(lambda x: len(str(x).split(" ")))
    ds_count = len(df.word_count)

    for i in range(0, ds_count):
        # Remove punctuation
        text = re.sub('[^a-zA-Z]', ' ', str(df['body'][i]))
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove tags
        text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
        
        # Remove special characters and digits
        text=re.sub("(\\d|\\W)+"," ",text)
        
        # Convert to list from string
        text = text.split()
        
        # Stemming
        ps=PorterStemmer()
        
        # Lemmatisation
        lem = WordNetLemmatizer()
        text = [lem.lemmatize(word) for word in text if not word in  
                stop_words] 
        text = " ".join(text)
        corpus.append(text)


    cv=CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,3))
    X=cv.fit_transform(corpus)

    return corpus

def most_common_keywords(corpus):

    # Convert most freq words to dataframe for plotting bar plot, save as CSV
    top_words = get_top_n_words(corpus, n=20)
    top_df = pd.DataFrame(top_words)
    top_df.columns=["Keyword", "Frequency"]

    return top_df

def most_common_bigrams(corpus):

    # Convert most freq bigrams to dataframe for plotting bar plot, save as CSV
    top2_words = get_top_n2_words(corpus, n=20)
    top2_df = pd.DataFrame(top2_words)
    top2_df.columns=["Bi-gram", "Frequency"]
    return top2_df

def most_common_trigrams(corpus):
    # Convert most freq trigrams to dataframe for plotting bar plot, save as CSV
    top3_words = get_top_n3_words(corpus, n=20)
    top3_df = pd.DataFrame(top3_words)
    top3_df.columns=["Tri-gram", "Frequency"]
    return top3_df

