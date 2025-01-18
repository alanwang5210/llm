from datasets import load_dataset
from transformers import AutoTokenizer


def add_prefix(example, prefix):
    example['instruction'] = prefix + example['instruction']
    return example


def preprocess_function(example):
    model_inputs = tokenizer(example["instruction"], max_length=512, truncation=True)
    labels = tokenizer(example["output"], max_length=32, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# 加载远程数据 https://huggingface.co/
# dataset = load_dataset("fka/awesome-chatgpt-prompts")

# 加载本地数据
# dataset = load_dataset("json", data_files="C:\\Users\\10100\\Downloads\\response.json")
# print (dataset['train'][0])

dataset = load_dataset("Chinese-Vicuna/guanaco_belle_merge_v1.0")
# 提取数据
data = dataset['train']
# 查看第一条具体数据
print(data[0])

# 数据集进行切分, train 训练集是90%，test集是10%,
# seed=0表示随机种子，保证每次切分都相同
dataset_split = data.train_test_split(test_size=0.95, seed=0)
print(dataset_split)

# select()方法挑选出其中的前两个数据
# tmp = dataset_split['train'].select([0, 1])
# print(tmp[0])

# filter()方法过滤出其中包含地球的数据
# tmp = dataset_split["train"].filter(lambda example: "地球" in example["input"])
# print(tmp)

# 传入自定义函数map()方法来实现数据映射
# prefix_dataset = dataset_split.map(lambda example: add_prefix(example, "Test +++++++"))
# print(prefix_dataset['train']['instruction'][1:5])

# 使用transformers库（Hugging Face推出的Python库，可以加载各种类型的Transformer模型）
# 加载bert-base-chinese模型提供的分词器，以便对原语料进行编码，将数据直接转换为模型可以接收的词向量形式
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
processed_dataset = dataset_split['train'].map(preprocess_function)
print(processed_dataset)
