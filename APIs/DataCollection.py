import re
import nltk
import json
import requests
import threading
import pandas as pd
import seaborn as sns
import plotly.express as px
from textblob import TextBlob
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from flask import Flask, render_template, Response, request, abort, send_from_directory

app = Flask(__name__)

@app.route('/collect', methods=['GET', 'POST'])
def collection(self, tup):
    self.reviews = []
    self.stars = []
    self.date = []
    self.country = []
    name, pages = tup
    for i in range(1, pages+1):
        page = requests.get(
            f"https://www.airlinequality.com/airline-reviews/{name}/page/{i}/?sortby=post_date%3ADesc&pagesize=100"
        )
        soup = BeautifulSoup(page.content, "html.parser")
        for item in soup.find_all("div", class_="text_content"):
            self.reviews.append(item.text)
        for item in soup.find_all("div", class_="rating-10"):
            try:
                self.stars.append(item.span.text)
            except:
                print(f"Error on page {i}")
        for item in soup.find_all("time"):
            self.date.append(item.text)
        for item in soup.find_all("h3"):
            self.country.append(item.span.next_sibling.text.strip(" ()"))
    self.stars = self.stars[:len(self.reviews)]
    self.df = pd.DataFrame(
        {"reviews": self.reviews, "stars": self.stars, "date": self.date, "country": self.country})
    return self.df
def __init__(self, df):
    self.df =  df
def cleaning(self):
    self.df['verified'] = self.df.reviews.str.contains("Trip Verified")
    lemma = WordNetLemmatizer()
    reviews_data = self.df.reviews.str.strip("✅ Trip Verified |")
    corpus = []
    for rev in reviews_data:
        rev = re.sub('[^a-zA-Z]', ' ', rev)
        rev = rev.lower()
        rev = rev.split()
        rev = [lemma.lemmatize(word) for word in rev if word not in set(stopwords.words("english"))]
        rev = " ".join(rev)
        corpus.append(rev)
    self.df['corpus'] = corpus
    self.df['date'] = self.df['date'].str.replace('st', '')
    self.df['date'] = self.df['date'].str.replace('nd', '')
    self.df['date'] = self.df['date'].str.replace('rd', '')
    self.df['date'] = self.df['date'].str.replace('th', '')
    self.df['date'] = self.df['date'].str.replace('Augu', 'August')
    self.df.date = pd.to_datetime(self.df.date)
    self.df.stars = self.df.stars.str.strip("\n\t\t\t\t\t\t\t\t\t\t\t\t\t")
    self.df.drop(self.df[self.df.stars == "None"].index, axis=0, inplace=True)
    self.df.drop(self.df[self.df.country.isnull() == True].index, axis=0, inplace=True)
    self.df.reset_index(drop=True)

def get_freq_dist(self, new_words, number_of_ngrams):
    from nltk import ngrams
    ngrams = ngrams(new_words, number_of_ngrams)
    ngram_fd = FreqDist(ngrams).most_common(40)
    ngram_sorted = {k: v for k, v in sorted(ngram_fd, key=lambda item: item[1])}
    ngram_joined = {'_'.join(k): v for k, v in sorted(ngram_fd, key=lambda item: item[1])}
    ngram_freqdist = pd.Series(ngram_joined)
    plt.figure(figsize=(10, 10))
    ax = ngram_freqdist.plot(kind="barh")
    plt.show()
    return ax

def star_val_count(self):
    self.df.stars.value_counts().plot(kind="bar")
    plt.xlabel("Ratings")
    plt.ylabel("Total Number of reviews with that rating")
    plt.suptitle("Counts for each ratings")
    plt.show()

def rating(self):
    df_ratings = pd.DataFrame(self.df.stars.value_counts())
    df_ratings = df_ratings.reset_index()
    pct_values = (df_ratings.stars.values / df_ratings.stars.values.sum() * 100).tolist()
    pct_values = [round(x, 2) for x in pct_values]
    df_ratings['pct_values'] = pct_values
    df_ratings = df_ratings.reset_index()
    df_ratings.rename(columns={'index': 'Stars', 'stars': 'total_counts'}, inplace=True)
    clrs = ['Red' if (x == max(df_ratings.total_counts)) else 'grey' for x in df_ratings.total_counts]
    ax = sns.barplot(x=df_ratings.Stars, y=df_ratings.total_counts, data=df_ratings, errwidth=0, palette=clrs)
    ax.bar_label(ax.containers[0])
    ax.set_xlabel("Ratings")
    ax.set_ylabel("Total Number of reviews with that rating")
    ax.set_title("Counts for each ratings")
    plt.show()

def country_review(self):
    df_country_review = pd.DataFrame(self.df.country.value_counts().head()).reset_index()
    df_country_review.rename(columns={'index': 'country', 'country': 'total_reviews'}, inplace=True)
    df_country_review.columns.values[0] = 'country'
    df_country_review.plot(kind="bar", x='country')
    plt.title("Maximum number of review by country")
    plt.show()

def country_rating(self):
    df_country_rating = pd.DataFrame(
        self.df.groupby('country').mean()['stars'].sort_values(ascending=False)).reset_index()
    df_country_rating.rename(columns={'stars': 'avg_rating'}, inplace=True)
    fig, ax = plt.subplots(figsize=(18, 5))
    ax1 = sns.barplot(x='country', y='avg_rating', data=df_country_rating[:12])
    ax.bar_label(ax.containers[0])
    ax.set_title("Top 12 Countries with avg highest rating")
    self.df.date = pd.to_datetime(self.df.date)
    fig = px.line(self.df, x='date', y="stars")
    fig.update_xaxes(rangeslider_visible=True)
    fig.show()

def word_map(self):
    plt.figure(figsize=(20, 10))
    stopwords = set(nltk.corpus.stopwords.words('english'))
    stopwords = text.ENGLISH_STOP_WORDS.union(['flight', "passenger", "u", "airway", "airline", "plane", "lhr", "review"])
    wordcloud = WordCloud(height=600, width=600, max_font_size=100, max_words=500, stopwords=stopwords).generate(
        " ".join(self.df.corpus))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

def word_count(self, new_words):
    nlp_words = FreqDist(new_words).most_common(20)
    all_fdist = pd.Series(dict(nlp_words))
    fig, ax = plt.subplots(figsize=(15, 8))
    all_plot = sns.barplot(x=all_fdist.index, y=all_fdist.values, ax=ax)
    all_plot.bar_label(all_plot.containers[0])
    plt.xticks(rotation=30)
    plt.show()

def word_analysis(self, new_words):
    self.get_freq_dist(new_words, 4)
    ratings_1_3 = self.df[self.df.stars.isin([1, 2, 3])]
    ratings_4_6 = self.df[self.df.stars.isin([4, 5, 6])]
    ratings_7_10 = self.df[self.df.stars.isin([7, 8, 9, 10])]
    reviews_1_3 = " ".join(ratings_1_3.corpus)
    reviews_4_6 = " ".join(ratings_4_6.corpus)
    reviews_7_10 = " ".join(ratings_7_10.corpus)
    words_1_3 = reviews_1_3.split(" ")
    words_4_6 = reviews_4_6.split(" ")
    words_7_10 = reviews_7_10.split(" ")
    new_words_1_3 = [word for word in words_1_3 if word not in set(nltk.corpus.stopwords.words('english'))]
    self.get_freq_dist(new_words_1_3, 4)
    new_words_4_6 = [word for word in words_4_6 if word not in set(nltk.corpus.stopwords.words('english'))]
    self.get_freq_dist(new_words_4_6, 4)
    new_words_7_10 = [word for word in words_7_10 if word not in set(nltk.corpus.stopwords.words('english'))]
    self.get_freq_dist(new_words_7_10, 4)

def polarity_scores(self):
    self.df['polarity'] = 0
    self.df['polarity'] = self.df['corpus'].apply(lambda x: TextBlob(x).sentiment.polarity)
    print(
        f"{self.df[(self.df['polarity'] > -0.2) & (self.df['polarity'] < 0.2)].shape[0]} number of reviews between -0.2 and 0.2 polarity score")
    print(
        f"{self.df[(self.df['polarity'] > -0.1) & (self.df['polarity'] < 0.1)].shape[0]} number of reviews between -0.1 and 0.1 polarity score")

def get_sentiment_label(corpus_text):
    vds = SentimentIntensityAnalyzer()
    score = vds.polarity_scores(corpus_text)['compound']
    if score > 0.2:
        return 1
    elif score < 0:
        return -1
    else:
        return 0

def sentiment_score(self):
    self.df['label'] = 0
    self.df['label'] = self.df['corpus'].apply(self.get_sentiment_label)

def topic_modelling(self):
    vect = CountVectorizer()
    tf = vect.fit_transform(self.df.corpus).toarray()
    tf_feature_names = vect.get_feature_names_out()
    number_of_topics = 8

def latent_allocation(self, number_of_topics, tf, tf_feature_names):
    model = LatentDirichletAllocation(n_components=number_of_topics, random_state=0)
    model.fit(tf)
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d words" % (topic_idx)] = ['{}'.format(tf_feature_names[i]) for i in
                                                      topic.argsort()[:-10 - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx)] = ['{:.1f}'.format(topic[i]) for i in
                                                        topic.argsort()[:-10 - 1:-1]]
    df_topic = pd.DataFrame(topic_dict)
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].barh(df_topic['Topic 0 words'], df_topic['Topic 0 weights'], color='skyblue')
    ax[0].set_title('Topic 0')
    ax[0].invert_yaxis()
    ax[1].barh(df_topic['Topic 1 words'], df_topic['Topic 1 weights'], color='salmon')
    ax[1].set_title('Topic 1')
    ax[1].invert_yaxis()
    plt.tight_layout()
    plt.show()

def nmf_topic_modelling(self, tf, tf_feature_names):
    nmf = NMF(n_components=2, init='random', random_state=0)
    nmf.fit_transform(tf)
    topic_dict = {}
    for topic_idx, topic in enumerate(nmf.components_):
        topic_dict["Topic %d words" % (topic_idx)] = ['{}'.format(tf_feature_names[i]) for i in
                                                      topic.argsort()[:-10 - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx)] = ['{:.1f}'.format(topic[i]) for i in
                                                        topic.argsort()[:-10 - 1:-1]]
    df_topic = pd.DataFrame(topic_dict)
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].barh(df_topic['Topic 0 words'], df_topic['Topic 0 weights'], color='skyblue')
    ax[0].set_title('Topic 0')
    ax[0].invert_yaxis()
    ax[1].barh(df_topic['Topic 1 words'], df_topic['Topic 1 weights'], color='salmon')
    ax[1].set_title('Topic 1')
    ax[1].invert_yaxis()
    plt.tight_layout()
    plt.show()

def eda(self):
    pass
    # reviews = " ".join(self.df.corpus)
    # words = reviews.split(" ")
    # new_words = [word for word in words if word not in set(nltk.corpus.stopwords.words('english'))]
    # self.word_count(new_words)
    # self.word_analysis(new_words)

if __name__ == '__main__':
    app.run(debug=True)