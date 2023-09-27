import re
import sqlite3
import threading
import pandas as pd
from bs4 import BeautifulSoup
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
import nltk
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction import text
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from flask import Flask, render_template, Response, request, abort, send_from_directory

app = Flask(__name__)

class SQLQueries:
    def __init__(self):
        self.conn = sqlite3.connect("static/database/info.db")
        self.curr = self.conn.cursor()
    def search_name(self, id):
        self.curr.execute('SELECT * FROM site_data WHERE id=?', (id,))
        results = self.curr.fetchall()
        return (results[0][1], results[0][2], results[0][3])
@app.errorhandler(404)
def page_not_found(e):
    return render_template(r"error404.html")

@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    id = ""
    if request.method == 'POST':
        id = request.form.get('id', '')
        if int(id)<=551:
            name, site_code, pages = SQLQueries().search_name(id)
            #threading.Thread(target=DataCollection().collection, args=[[site_code, pages]]).start()
            return render_template('analysis.html', id=name)
        else:
            abort(404)
    else:
        abort(404)

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template(r"homepage.html")

if __name__ == '__main__':
    app.run(debug=True)