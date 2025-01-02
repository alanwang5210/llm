import pandas as pd
from nltk.corpus import stopwords

from textblob import TextBlob


# 计算平均单词长度
def avg_word(sentence):
    words = sentence.split()
    return sum(len(word) for word in words) / len(words)


data = pd.read_csv("tweets_data.csv")

# 通过lower()方法可以将数据集中的英文大写字母转换为英文小写字母。通过upper()方法可以将数据集中的英文小写字母转换为英文大写字母
data['text'] = data['text'].apply(lambda sen: " ".join(x.lower() for x in sen.split()))

# 由于标点符号、特殊符号在文本数据中不表示任何额外的信息，因此去除标点符号有助于减小训练数据的规模。
# 这里通过replace()方法去除字符串中的非字母数字字符（如标点符号、特殊符号等），也可以通过Re库进行处理。
data['text'] = data['text'].str.replace(r'[^\w\s]', '', regex=True)

# 去除常用停用词
#在某些任务中需要从文本数据中去除停用词（或常见单词）。可以创建停用词列表或使用预定义的库，逐一过滤文本中与停用词列表匹配的项。
# 这里以NLTK库提供的停用词列表为例进行测试
stop = stopwords.words('english')
data['text'] = data['text'].apply(lambda sen: " ".join(x for x in sen.split() if x not in stop))

#稀缺词是指在文本数据中较少出现，并且采用不常用表达方式的词语。我们可以首先对稀缺词进行统计，然后将其直接删除或替换为常见的表达方式。
freq = pd.Series(' '.join(data['text']).split()).value_counts()[-10:]
data['text'] = data['text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

#由于以互联网社交媒体为来源的数据存在大量的拼写错误，因此，拼写校正是一个十分重要的预处理步骤。此处选用TextBlob库进行处理。
# TextBlob库可以用来处理多种自然语言处理任务，如词性标注、名词性成分提取、情感分析、文本翻译等。
data['text'] = data['text'].apply(lambda x: str(TextBlob(x).correct()))

# 可以提取每个推文数据的基本特征之一——单词数量。借助split()方法将句子按空格进行切分，并对单词数量进行计数。
data['word_count'] = data['text'].apply(lambda x: len(str(x).split(' ')))

# 将每个推文数据的字符数量（每条推文数据包含的字母数）除以单词数量，即可得到平均单词长度
data['avg_word'] = data['text'].apply(lambda x: avg_word(x))
print(data[['text', 'avg_word']])