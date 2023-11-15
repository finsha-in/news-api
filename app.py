from flask import Flask, request, jsonify
import requests
from datetime import datetime
import re
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


subscription_key = "07068875780c4a74bad61bddbede2826"

app = Flask(__name__)

def clean_summary(text):
    cleaned_text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
    return cleaned_text.strip()

def generate_summary(news, summary_proportion=0.1):
    # Extracting headlines from the news articles
    headlines = [article['name'] for article in news['value']]
    combined_titles = " ".join(headlines)

    # Function to split text into sentences
    def split_into_sentences(text):
        sentences = sent_tokenize(text)
        return sentences

    sentences = split_into_sentences(combined_titles)
    num_sentences = len(sentences)
    num_summary_sentences = max(int(num_sentences * summary_proportion), 1)  # Ensure at least one sentence

    # Calculating TF-IDF and ranking sentences
    tfidf_vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
    top_sentence_indices = np.argsort(sentence_scores)[-num_summary_sentences:]
    summary = [sentences[i] for i in sorted(top_sentence_indices)]

    # Joining the top sentences to form a summary
    return ' '.join(summary)

def get_news_and_summary(company_name):
    endpoint = f"https://api.bing.microsoft.com/v7.0/news/search?q={company_name}&category=Business&count=200&offset=0&mkt=en-in&safeSearch=Moderate&textFormat=Raw&textDecorations=false"
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    response = requests.get(endpoint, headers=headers)
    news = response.json()

    allowed_websites = ["moneycontrol.com", "cnbctv18.com", "livemint.com", "ndtv.com", "economictimes.indiatimes.com",
                        "business-standard.com", "reuters.com", "bloomberg.com", "marketwatch.com", "finance.yahoo.com",
                        "forbes.com", "moneycrashers.com", "investors.com", "investing.com", "reuters.com",
                        "mtnewswires.com", "djnewswires.com"]

    relevant_articles = [
        article for article in news['value']
        if any(website in article['url'] for website in allowed_websites)
    ]

    final_articles = []

    for article in relevant_articles:
        if ('about' in article and any('share price' in about['name'].lower() for about in article['about'])) \
                or 'price' in article['name'].lower():
            continue

        if len(article['name'].split()) < 8:
            continue

        article_info = {
            'headline': article['name'],
            'url': article['url'],
            'datePosted': article['datePublished']
        }

        if 'image' in article and 'thumbnail' in article['image']:
            article_info['thumbnail'] = article['image']['thumbnail']['contentUrl']

        final_articles.append(article_info)

    formatted_articles = []
    for article in final_articles:
        date_published = datetime.strptime(article['datePosted'], '%Y-%m-%dT%H:%M:%S.%f0Z')
        article['date'] = date_published.strftime('%Y-%m-%d')
        article['time'] = date_published.strftime('%H:%M')
        del article['datePosted']
        formatted_articles.append(article)

    summary = generate_summary(news)

    return formatted_articles, summary

@app.route('/news', methods=['GET'])
def news():
    company_name = request.args.get('company_name')

    if not company_name:
        return jsonify({'error': "Company name not provided!"}), 400

    try:
        formatted_articles, summary = get_news_and_summary(company_name)
        return jsonify({'articles': formatted_articles, 'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return jsonify({'message': 'Welcome to the News Summary API'})

if __name__ == "__main__":
    app.run(debug=True)
