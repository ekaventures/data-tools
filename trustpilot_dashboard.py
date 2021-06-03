from hamish_scraper import scrape_trustpilot_reviews
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
from keyword_analyser import analyse_words, most_common_bigrams, most_common_keywords, most_common_trigrams

#Function to download results

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

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
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
#st.sidebar.write("Choose the reviews you would like to analyse for Section 3")
#reviews_of_interest = st.sidebar.multiselect("Rewiews to analyse:",('1','2','3','4','5'),['1','2','3','4','5'])

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
    
    #dataset1 = df2['rating'].astype(str)
    #dataset2 = df2[df2['rating'].isin(reviews_of_interest)]
    #dataset = dataset2[['created_at','body']]
    #dataset = dataset[(dataset['body']!="")]
    df_words = df2[['created_at','body']]



    if st.button('Download Specific Reviews'):
        tmp_download_link = download_link(dataset, 'Trustpilot_reviews4.csv', 'Click here to download your data!')
        st.markdown(tmp_download_link, unsafe_allow_html=True)


    st.subheader("Section 3: Trustpilot Review Themes")

    words = analyse_words(df_words)

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
