import nltk
import numpy as np

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup

lemmatizer = WordNetLemmatizer()

with open("stopwords.txt") as f:
    stopwords = set(w.rstrip for w in f)

print(stopwords[:10])

positive_reviews = BeautifulSoup(open("eletronics/positive.reviews").read()).findAll("review_text")
negative_reviews = BeautifulSoup(open("eletronics/negative.reviews").read()).findAll("review_text")

# lets balance the dataset because one group is much higher than the other
np.random.shuffle(positive_reviews)
positive_reviews = positive_reviews[:len(negative_reviews)]

# now we need to create a global index for each word
# we need to find the size of vocabulary
word_index_map = {}
current_index = 0

def my_tokenizer(sentence:str):
    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords]

    return tokens

for review in positive_reviews:
    tokens = my_tokenizer(review)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1
