#Sentiment Analysis Setup#

from datetime import date
import snscrape.modules.twitter as twitter
import pandas as pd
import numpy as np
from textblob import TextBlob
from wordcloud import WordCloud
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import spacy
nlp = spacy.load("en_core_web_sm")

queries = ['residential proxies']

max_results = 100

def scrape_search(query):
    scraper = twitter.TwitterSearchScraper(query)
    return scraper
    
for query in queries:
    output_filename = query.replace(" ", "_") + ".txt" 
    with open(output_filename, 'w') as f:
        scraper = scrape_search(query)
        i = 0
        for i, tweet in enumerate(scraper.get_items(), start = 1):

                        tweet_json = json.loads(tweet.json())

print (f"\nScraped tweet: {tweet_json['content']}")

f.write(tweet.json())
f.write('\n')
f.flush()