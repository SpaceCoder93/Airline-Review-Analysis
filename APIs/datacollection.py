import re
import nltk
import requests
import threading
import pandas as pd
import seaborn as sns
import plotly.express as px
from textblob import TextBlob
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from flask import Flask, jsonify
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation, NMF

app = Flask(__name__)

@app.route('/collect', methods=['GET', 'POST'])
def collection( tup):
    reviews = []
    stars = []
    date = []
    country = []
    name, pages = tup
    for i in range(1, pages+1):
        page = requests.get(
            f"https://www.airlinequality.com/airline-reviews/{name}/page/{i}/?sortby=post_date%3ADesc&pagesize=100"
        )
        soup = BeautifulSoup(page.content, "html.parser")
        for item in soup.find_all("div", class_="text_content"):
            reviews.append(item.text)
        for item in soup.find_all("div", class_="rating-10"):
            try:
                stars.append(item.span.text)
            except:
                print(f"Error on page {i}")
        for item in soup.find_all("time"):
            date.append(item.text)
        for item in soup.find_all("h3"):
            country.append(item.span.next_sibling.text.strip(" ()"))
    stars = stars[:len(reviews)]
    df = pd.DataFrame(
        {"reviews": reviews, "stars": stars, "date": date, "country": country})
    return jsonify(df.to_dict(orient='records'))

def cleaning(df):
    df['verified'] = df.reviews.str.contains("Trip Verified")
    lemma = WordNetLemmatizer()
    reviews_data = df.reviews.str.strip("✅ Trip Verified |")
    corpus = []
    for rev in reviews_data:
        rev = re.sub('[^a-zA-Z]', ' ', rev)
        rev = rev.lower()
        rev = rev.split()
        rev = [lemma.lemmatize(word) for word in rev if word not in set(stopwords.words("english"))]
        rev = " ".join(rev)
        corpus.append(rev)
    df['corpus'] = corpus
    df['date'] = df['date'].str.replace('st', '')
    df['date'] = df['date'].str.replace('nd', '')
    df['date'] = df['date'].str.replace('rd', '')
    df['date'] = df['date'].str.replace('th', '')
    df['date'] = df['date'].str.replace('Augu', 'August')
    df.date = pd.to_datetime(df.date)
    df.stars = df.stars.str.strip("\n\t\t\t\t\t\t\t\t\t\t\t\t\t")
    df.drop(df[df.stars == "None"].index, axis=0, inplace=True)
    df.drop(df[df.country.isnull() == True].index, axis=0, inplace=True)
    df.reset_index(drop=True)

def get_freq_dist( new_words, number_of_ngrams):
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

def star_val_count(df):
    df.stars.value_counts().plot(kind="bar")
    plt.xlabel("Ratings")
    plt.ylabel("Total Number of reviews with that rating")
    plt.suptitle("Counts for each ratings")
    plt.show()

def rating(df):
    df_ratings = pd.DataFrame(df.stars.value_counts())
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

def country_review(df):
    df_country_review = pd.DataFrame(df.country.value_counts().head()).reset_index()
    df_country_review.rename(columns={'index': 'country', 'country': 'total_reviews'}, inplace=True)
    df_country_review.columns.values[0] = 'country'
    df_country_review.plot(kind="bar", x='country')
    plt.title("Maximum number of review by country")
    plt.show()

def country_rating(df):
    df_country_rating = pd.DataFrame(
        df.groupby('country').mean()['stars'].sort_values(ascending=False)).reset_index()
    df_country_rating.rename(columns={'stars': 'avg_rating'}, inplace=True)
    fig, ax = plt.subplots(figsize=(18, 5))
    ax1 = sns.barplot(x='country', y='avg_rating', data=df_country_rating[:12])
    ax.bar_label(ax.containers[0])
    ax.set_title("Top 12 Countries with avg highest rating")
    df.date = pd.to_datetime(df.date)
    fig = px.line(df, x='date', y="stars")
    fig.update_xaxes(rangeslider_visible=True)
    fig.show()

def word_map(df):
    plt.figure(figsize=(20, 10))
    stopwords = set(nltk.corpus.stopwords.words('english'))
    stopwords = text.ENGLISH_STOP_WORDS.union(['flight', "passenger", "u", "airway", "airline", "plane", "lhr", "review"])
    wordcloud = WordCloud(height=600, width=600, max_font_size=100, max_words=500, stopwords=stopwords).generate(
        " ".join(df.corpus))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

def word_count( new_words):
    nlp_words = FreqDist(new_words).most_common(20)
    all_fdist = pd.Series(dict(nlp_words))
    fig, ax = plt.subplots(figsize=(15, 8))
    all_plot = sns.barplot(x=all_fdist.index, y=all_fdist.values, ax=ax)
    all_plot.bar_label(all_plot.containers[0])
    plt.xticks(rotation=30)
    plt.show()

def word_analysis(df, new_words):
    get_freq_dist(new_words, 4)
    ratings_1_3 = df[df.stars.isin([1, 2, 3])]
    ratings_4_6 = df[df.stars.isin([4, 5, 6])]
    ratings_7_10 = df[df.stars.isin([7, 8, 9, 10])]
    reviews_1_3 = " ".join(ratings_1_3.corpus)
    reviews_4_6 = " ".join(ratings_4_6.corpus)
    reviews_7_10 = " ".join(ratings_7_10.corpus)
    words_1_3 = reviews_1_3.split(" ")
    words_4_6 = reviews_4_6.split(" ")
    words_7_10 = reviews_7_10.split(" ")
    new_words_1_3 = [word for word in words_1_3 if word not in set(nltk.corpus.stopwords.words('english'))]
    get_freq_dist(new_words_1_3, 4)
    new_words_4_6 = [word for word in words_4_6 if word not in set(nltk.corpus.stopwords.words('english'))]
    get_freq_dist(new_words_4_6, 4)
    new_words_7_10 = [word for word in words_7_10 if word not in set(nltk.corpus.stopwords.words('english'))]
    get_freq_dist(new_words_7_10, 4)

def polarity_scores(df):
    df['polarity'] = 0
    df['polarity'] = df['corpus'].apply(lambda x: TextBlob(x).sentiment.polarity)
    print(
        f"{df[(df['polarity'] > -0.2) & (df['polarity'] < 0.2)].shape[0]} number of reviews between -0.2 and 0.2 polarity score")
    print(
        f"{df[(df['polarity'] > -0.1) & (df['polarity'] < 0.1)].shape[0]} number of reviews between -0.1 and 0.1 polarity score")

def get_sentiment_label(corpus_text):
    vds = SentimentIntensityAnalyzer()
    score = vds.polarity_scores(corpus_text)['compound']
    if score > 0.2:
        return 1
    elif score < 0:
        return -1
    else:
        return 0

def sentiment_score(df):
    df['label'] = 0
    df['label'] = df['corpus'].apply(get_sentiment_label)

def topic_modelling(df):
    vect = CountVectorizer()
    tf = vect.fit_transform(df.corpus).toarray()
    tf_feature_names = vect.get_feature_names_out()
    number_of_topics = 8

def latent_allocation( number_of_topics, tf, tf_feature_names):
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

def nmf_topic_modelling( tf, tf_feature_names):
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

def eda():
    pass
    # reviews = " ".join(df.corpus)
    # words = reviews.split(" ")
    # new_words = [word for word in words if word not in set(nltk.corpus.stopwords.words('english'))]
    # word_count(new_words)
    # word_analysis(new_words)

if __name__ == '__main__':
    app.run(port=5001, debug=True)