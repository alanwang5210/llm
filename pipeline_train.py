import os

#os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_ENDPOINT'] = 'https://mirrors.tuna.tsinghua.edu.cn/hugging-face'

from transformers import pipeline

# 下载并缓存一个默认的预训练模型和分词器, 增加参数model来指定模型
classifier = pipeline("sentiment-analysis", model="IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment")

print(classifier(["I played very poorly in today's game and was blamed by the coach.",
                  "It was the greatest luck of my life to meet her, and I wish her a happy birthday!"]))
