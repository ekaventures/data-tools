import streamlit as st
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
import plotly.express as px
import base64
import requests
import math
from bs4 import BeautifulSoup
import random
import time
import json
import numpy as np
import re
import nltk
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

# Analyse the words

def analyse_words(df_input):

    corpus = []
    body_df = df_input.reset_index(drop = True)
    body_df['word_count'] = body_df['body'].apply(lambda x: len(str(x).split(" ")))
    ds_count = len(body_df.word_count)

    for i in range(0, ds_count):

        # Remove punctuation
        text = re.sub('[^a-zA-Z]', ' ', str(body_df['body'][i]))
        
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

# Convert most freq words to dataframe for plotting bar plot

def most_common_keywords(corpus):

    top_words = get_top_n_words(corpus, n=20)
    top_df = pd.DataFrame(top_words)
    top_df.columns=["Keyword", "Frequency"]

    return top_df

# Convert most freq bigrams to dataframe for plotting bar plot

def most_common_bigrams(corpus):

    top2_words = get_top_n2_words(corpus, n=20)
    top2_df = pd.DataFrame(top2_words)
    top2_df.columns=["Bi-gram", "Frequency"]
    return top2_df

# Convert most freq trigrams to dataframe for plotting bar plot

def most_common_trigrams(corpus):
    
    top3_words = get_top_n3_words(corpus, n=20)
    top3_df = pd.DataFrame(top3_words)
    top3_df.columns=["Tri-gram", "Frequency"]
    return top3_df

#Function to download results as a CSV

def download_link(object_to_download, download_filename, download_link_text):

    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

#Function to sleep for a random time

def sleep_random(x,y):
    time.sleep(random.randint(x,y))

#Scrape reviews function 

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def scrape_trustpilot_reviews(url):

    # Trustpilot review page
    reviewPage = url

    # Trustpilot default 
    resultsPerPage = 20 

    # Throttling to avoid spamming page with requests
    # With sleepTime seconds between every page request
    throttle = True
    sleepTime = 1

    # Find number of pages
    req = requests.get(reviewPage).text
    soup = BeautifulSoup(req,'html.parser')
    reviews = soup.find("span", class_ = "headline__review-count").text.replace(',','')
    review_count = int(reviews)
    #st.write('Company has total reviews of ',review_count)
    pages = math.ceil(review_count / resultsPerPage)
    #st.write('Found total of ' + str(pages) + ' pages to scrape')

    #Create base dataframe
    data = {'title': [], 'body': [],'rating':[],'created_at':[]}
    df = pd.DataFrame.from_dict(data)

    my_bar = st.progress(0.0)

    #Loop through pages
    for i in range(1,pages+2):

        percent_complete = (i)/pages
        my_bar.progress(percent_complete)

        if(throttle): time.sleep(sleepTime)

        if i == 1: 

            req = requests.get(reviewPage).text
            soup = BeautifulSoup(req,'html.parser')
            review_body = soup.find_all("div",class_ = "review-content")
            next_page_button = soup.find("nav",class_ = 'pagination-container AjaxPager')
            next_page_url = 'https://uk.trustpilot.com/' + next_page_button.a['href']

        else:

            req = requests.get(next_page_url).text
            soup = BeautifulSoup(req,'html.parser')
            review_body = soup.find_all("div",class_ = "review-content")
            if soup.find("a",class_ = 'button button--primary next-page') is not None:

                next_page_button = soup.find("a",class_ = 'button button--primary next-page')
                next_page_url = 'https://uk.trustpilot.com/' + next_page_button['href']

            else: 

                break


        for j in range(0,19):


            stars = review_body[j].img['alt'].strip().split(' ')[0]
            heading = review_body[j].a.text.strip()
            if review_body[j].p is not None:
                body = review_body[j].p.text.strip()
            else:
                body = ''
            json_object = json.loads(review_body[j].script.contents[0])
            review_date = json_object['publishedDate']
            new_row = {'title':heading, 'body':body, 'rating':stars, 'created_at':review_date}
            df = df.append(new_row,ignore_index = True)


    return df 


st.set_page_config(layout="wide")
st.title("Trustpilot Dashboard")
st.write("This tool fetches Trustpilot reviews and runs some basic analysis")
st.write("Developed by [Eka Ventures](https://www.ekavc.com/)")

#Collect desired page to analyse
st.sidebar.write("Enter the Trustpilot page to analyse")
url = st.sidebar.text_input('Input Trustpilot URL:')
run_button = st.sidebar.checkbox("Fetch Trustpilot Reviews")
st.sidebar.write("Choose the reviews you would like to analyse for Section 3")
reviews_of_interest = st.sidebar.multiselect("Rewiews to analyse:",('1','2','3','4','5'),['1','2','3','4','5'])

if run_button == True:

    df = scrape_trustpilot_reviews(url)
    df2 = df


    df2['date'] = df2['created_at'].apply(lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'))
    df2['day'] = pd.to_datetime(df2['date'])
    df2["Day_of_Week"] = df2.day.dt.weekday
    df2["created_at_week"] = df2.apply(lambda x: x['day'] - timedelta(days=x['Day_of_Week']), axis=1).dt.date
    df2['month'] = df2['day'].to_numpy().astype('datetime64[M]')
    df2['created_at_month'] = df2['month'].dt.date
    df_download = df2[['date','rating','title','body']]


    st.subheader("Section 1: Trustpilot Reviews")
    st.write('Below is a sample of the results found:')
    st.write(df_download.head(10))

    #df_final = df[["day","created_at_week","created_at_month","title","body","rating"]]

    if st.button('Download Trustpilot Reviews'):
        tmp_download_link = download_link(df_download, 'Trustpilot_reviews.csv', 'Click here to download your data!')
        st.markdown(tmp_download_link, unsafe_allow_html=True)

    #df.groupby(["created_at_month", "rating"]).size()

    count_of_reviews = df2.groupby(["created_at_month", "rating"]).size().reset_index(name="review_count")
    count_of_reviews_2 = count_of_reviews 
    count_of_reviews_2['rating'] = count_of_reviews_2['rating'].astype(str)

    st.subheader("Section 2: Trending Trustpilot Reviews")

    fig = px.bar(count_of_reviews_2, 
                x="created_at_month", 
                y="review_count", 
                color="rating", 
                title="Trustpilot Review Count",
                labels={
                        "created_at_month": "Month",
                        "review_count": "Reviews",
                        "rating": "Rating"
                    },
                color_continuous_scale='Bluered_r',
                category_orders={"rating": ['5','4','3','2','1']},
                width = 1280,
                height = 600
    )

    st.plotly_chart(fig)

    df3 = count_of_reviews.groupby(['created_at_month'])['review_count'].agg('sum')
    df4 = count_of_reviews.merge(df3, left_on = 'created_at_month',right_on = 'created_at_month')
    df4.rename(columns={'review_count_x':'review_count','review_count_y':'total'}, inplace=True)
    df4.sort_values(by=['created_at_month'])
    df4['proportion'] = (100*df4['review_count']/df4['total']).round(1)

    fig2 = px.line(df4,
                x="created_at_month", 
                y="proportion", 
                title = 'Proportion of Reviews by Rating',
                color='rating',
                width = 1280,
                height = 600,
                category_orders={"rating": ['5','4','3','2','1']},
                labels={
                        "created_at_month": "Month",
                        "proportion": "Percentage",
                        "rating": "Rating"
                    }
                    )
    st.plotly_chart(fig2)
    
    dataset1 = df_download['rating'].astype(str)
    dataset2 = df_download[df_download['rating'].isin(reviews_of_interest)]
    dataset = dataset2[['date','body']]

    st.subheader("Section 3: Trustpilot Review Themes")

    words = analyse_words(dataset)
    keywords = most_common_keywords(words)
    bigrams = most_common_bigrams(words)
    trigrams = most_common_trigrams(words)



    fig3 = px.bar(keywords, 
                    x='Keyword', 
                    y='Frequency',
                    title = 'Most Common Keywords',
                    width = 1280,
                    height = 600,)

    st.plotly_chart(fig3)

    fig4 = px.bar(bigrams, 
                    x='Bi-gram', 
                    y='Frequency',
                    title = 'Most Common Bi-Grams',
                    width = 1280,
                    height = 600,)

    st.plotly_chart(fig4)

    fig5 = px.bar(trigrams, 
                    x='Tri-gram', 
                    y='Frequency',
                    title = 'Most Common Tri-Grams',
                    width = 1280,
                    height = 600,)

    st.plotly_chart(fig5)

