import pandas as pd
from nltk.corpus import stopwords


def avg_word(sentence):
    words = sentence.split()
    return sum(len(word) for word in words) / len(words)


data = pd.read_csv("tweets_data.csv")

stop = stopwords.words('english')

data['text'] = data['text'].apply(lambda sen: " ".join(x for x in sen.split() if x not in stop))

data['text'] = data['text'].str.replace(r'[^\w\s]', '', regex=True)

data['word_count'] = data['text'].apply(lambda x: len(str(x).split(' ')))

data['avg_word'] = data['text'].apply(lambda x: avg_word(x))
print(data[['text', 'avg_word']])
