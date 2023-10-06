import json
# load pkgs
import pandas as pd
import numpy as np

# text cleaning
import neattext.functions as nfx
import neattext as nt
import re
# keyword extract
from collections import Counter
# sentiment analysis(失敗)
from textblob import TextBlob

# Load ML Pkgs
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
# Vectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# Metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix
# Split Our Dataset
from sklearn.model_selection import train_test_split

import sys
from google_trans_new import google_translator

with open("train.json", 'r', encoding="UTF-8") as f:
    train_data = json.load(f)
    train_data = pd.DataFrame(train_data)

with open("test.json", 'r', encoding="UTF-8") as f:
    test_data = json.load(f)
    test_data = pd.DataFrame(test_data)


def get_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        result = "positive"
    elif sentiment < 0:
        result = "negative"
    else:
        result = "neutral"
    return result


def single(text):
    result = ""
    blob = TextBlob(text)
    token = blob.words
    for w in token:
        w = w.singularize()
        result += w
        result += " "
    return result


# 去掉括号及其里面的内容
def remove_bracket_text(text):
    result = re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]", " ", text)
    return result


def remove_colon(text):
    result = re.sub(":", "", text)
    # result = result.replace("(", " ")
    # result = result.replace(")", " ")
    result = result.replace("|", " ")
    return result


# pic.twitter.com/OxJVb4rs9T
def extra_remove(text):
    result = text.replace("https://t.co/aLtFLYVOpe pic.twitter.com/BGJtknBCu3", " ")
    result = re.sub(r'pic.twitter.com/[0-9a-zA-Z]+', ' ', result)
    result = re.sub("Macau|Macao", "Macau", result)
    result = re.sub("Hong Kong", "HongKong", result)
    result = re.sub("U.S.|USA|United States", "US", result)
    result = re.sub("Cuvid-19", "Covid-19", result)
    result = re.sub("February", "Feb", result)
    result = re.sub("January", "Jan", result)
    result = re.sub("The World Health Organization", "WHO", result)
    return result


def str_lower(text):
    result = text.lower()
    return result


def delete_http(text):
    result = re.sub(r'tmsnrtrs/[0-9a-zA-Z]*', ' ', text)
    return result


def extract_keywords(text, num=50):
    tokens = [tok for tok in text.split()]
    most_common_tokens = Counter(tokens).most_common(num)
    return dict(most_common_tokens)


def request(text):
    lang = "en"
    t = google_translator()
    translate_text = t.translate(text.strip(), lang)
    return translate_text


train_data['sentiment'] = train_data['content'].apply(get_sentiment)
# print(data.head)

# compare our label vs sentiment
group = train_data.groupby(['label', 'sentiment']).size()


# print(group)

def cleaning_data(data):
    data['clean_content'] = data['content'].apply(nfx.remove_emails)
    data['clean_content'] = data['clean_content'].apply(nfx.remove_urls)
    data['clean_content'] = data['clean_content'].apply(extra_remove)
    data['clean_content'] = data['clean_content'].apply(nfx.remove_hashtags)
    data['clean_content'] = data['clean_content'].apply(nfx.remove_userhandles)
    data['clean_content'] = data['clean_content'].apply(nfx.remove_punctuations)
    data['clean_content'] = data['clean_content'].apply(nfx.remove_bad_quotes)
    data['clean_content'] = data['clean_content'].apply(remove_colon)
    data['clean_content'] = data['clean_content'].apply(remove_bracket_text)
    data['clean_content'] = data['clean_content'].apply(delete_http)
    data['clean_content'] = data['clean_content'].apply(nfx.remove_stopwords)
    data['clean_content'] = data['clean_content'].apply(nfx.remove_multiple_spaces)
    return data['content']


train_data['clean_content'] = cleaning_data(train_data)
test_data['clean_content'] = cleaning_data(test_data)

zero_list = train_data[train_data['label'] == 0]['clean_content'].tolist()
zero_docx = ' '.join(zero_list)
keyword_zero = extract_keywords(zero_docx)

one_list = train_data[train_data['label'] == 1]['clean_content'].tolist()
one_docx = ' '.join(one_list)
keyword_one = extract_keywords(one_docx)
# print("标签为1的关键字:　",keyword_one)
# print("标签为0的关键字:　",keyword_zero)


# machine learning
Xfeatures = train_data['clean_content']
ylabels = train_data['label']

cv = CountVectorizer(min_df=0, lowercase=True, encoding='utf-8')
X = cv.fit_transform(Xfeatures)

# print(cv.vocabulary_)
#
# print(X.toarray())

tfidf = TfidfTransformer()

X = tfidf.fit_transform(X)

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.1, random_state=1000, stratify=ylabels)

# model
nv_model = MultinomialNB(alpha=0.01, class_prior=[0.1, 0.3])
# nv_model.fit(X_train, y_train)
nv_model.fit(X, ylabels)


# LogisticRegression
# lr_model = LogisticRegression(solver='lbfgs', max_iter=300)
# lr_model.fit(X,ylabels)
#
# sv_model = SVC()
# sv_model.fit(X_train,y_train)

# accuracy
# score_nv = nv_model.score(X_test, y_test)
# score_lr = lr_model.score(X_test, y_test)
# score_sv = sv_model.score(X_test, y_test)
# print("MultinomialNB:", score_nv)
# print("LogisticRegression: ",score_lr)
# print("SVM: ",score_sv)
# prediction
# y_pred_for_nv = nv_model.predict(X_test)
# y_pred_for_lr = lr_model.predict(X_test)


# make a single prediction
def predict_label(sample_text, model):
    sample_text = [sample_text]
    myvect = cv.transform(sample_text)
    myvect = tfidf.transform(myvect).toarray()
    prediction = model.predict(myvect)
    print(prediction[0])


with open("test_result.txt", 'w') as sys.stdout:
    for test in test_data['clean_content']:
        predict_label(test, nv_model)
