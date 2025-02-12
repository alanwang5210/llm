from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

# 加载本地
# model_name_or_path = "D:\\workspace\\alan\\pyRag\\models\\BAAI\\bge-m3"
# model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


model_name_or_path = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
# 加载模型
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# encoded_input = tokenizer("It is been particularly cold today")

# 每个句子的长度不等，但模型的输入一般需要统一的形状，因此可以通过填充的方法向Token
# 较少的句子添加特殊的填充Token以进行占位。操作方法是向tokenizer()方法传入参数padding=True
#encoded_input = tokenizer(["It is been particularly cold today", "This movie is good"], padding=True)

#句子太长时，我们可以采取相反的做法，对过长的句子进行截断。操作方法是向tokenizer()方法传入参数truncation=True
#encoded_input = tokenizer(["It is been particularly cold today", "This movie is good"], truncation=True)

#若要指定返回的张量类型（如用于不同程序的PyTorch、TensorFlow或NumPy数据类型）
# ，可以在tokenizer()方法中增加参数return_tensors。
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
encoded_input = tokenizer(raw_inputs,
                          padding=True,
                          truncation=True,
                          return_tensors="pt")
print(encoded_input)


# input_ids对应句子中每个Token的索引；token_type_ids用于标识在多序列情况下Token的所属序列；
# attention_mask表明Token是否需要被注意（1表示被注意，0表示不需要被注意）
print(encoded_input)

# 使用decode()方法将索引(input_ids)序列解码为原始输入
# [CLS]和[SEP]标识符是BERT模型中的特殊Token。[CLS]置于句子的首位，
# 表示经过BERT模型处理得到的表征向量可以用于后续的分类任务，[SEP]用于分隔两个句子
print(tokenizer.decode(encoded_input["input_ids"]))
