import os
import time

from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

#全参数微调
#更新预训练模型所有权重参数（所有层、所有神经元）。

# 记录开始时间，用于计算整个脚本运行的时间
start_time = time.time()

# 设置环境变量 HF_ENDPOINT 为镜像站点地址，以加速模型和数据集的下载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 加载指定的数据集，并创建一个子集（前200个样本）以便训练和测试
raw_datasets = load_dataset("Chinese-Vicuna/guanaco_belle_merge_v1.0")

print(raw_datasets)
data = Dataset.from_dict(raw_datasets['train'][:200])  # 增加样本量到200
# 将数据集划分为训练集和测试集，使用42作为随机种子保证可重复性
dataset_split = data.train_test_split(test_size=0.5, seed=42)

# 定义使用的预训练模型名称或路径及标签数量
model_name_or_path = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)  # 加载预训练的分词器
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=2)  # 加载预训练模型并设置类别数


# 定义文本tokenize函数，将输入文本转化为模型输入格式
def tokenize_function(examples):
    # Tokenize the input texts
    tokenized = tokenizer(examples["instruction"], padding="max_length", truncation=True)

    # Handle labels: Ensure that each example has a single label (integer)
    if "label" in examples:
        labels = []
        for label in examples["label"]:
            # 确保每个label都是整数
            if isinstance(label, int):
                labels.append(label)
            else:
                raise ValueError(f"Unexpected label type: {type(label)}. Expected type: int")
        tokenized['labels'] = labels
    else:

        # for label in examples["instruction"]:
        #     # 确保每个label都是整数
        #     if isinstance(label, int):
        #         labels.append(1)
        labels = [1 for label in examples["instruction"]]
        tokenized['labels'] = labels
    return tokenized


# 对数据集应用tokenize函数
tokenized_datasets = dataset_split.map(tokenize_function, batched=True)

# 随机打乱数据集，并选择一部分数据作为训练集和评估集
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))  # 从200中选100作为训练集
eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(100))  # 同样地，选取100个样本作为验证集

# 初始化DataCollator，用于在批处理时对数据进行填充
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# 定义评估指标函数，计算准确率、精确率、召回率和F1分数
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',  # 模型保存目录
    eval_strategy="epoch",  # 每个epoch结束后进行评估
    per_device_train_batch_size=16,  # 训练时每个设备上的batch大小
    per_device_eval_batch_size=16,  # 评估时每个设备上的batch大小
    learning_rate=3e-5,  # 学习率
    num_train_epochs=3,  # 总共训练的epochs数
    warmup_ratio=0.1,  # warm up步数占总步数的比例
    logging_dir='./logs',  # 日志保存目录
    logging_steps=10,  # 每多少步记录一次日志
    save_strategy="epoch",  # 每个epoch结束时保存模型
    logging_strategy="epoch",
    load_best_model_at_end=True,  # 训练结束时加载最优模型
    metric_for_best_model="accuracy",  # 使用准确率来选择最优模型
    report_to="tensorboard")

# 初始化Trainer对象
trainer = Trainer(
    model=model,  # 要训练的模型
    args=training_args,  # 训练参数
    train_dataset=train_dataset,  # 训练数据集
    eval_dataset=eval_dataset,  # 验证数据集
    tokenizer=tokenizer,  # 分词器
    data_collator=data_collator,  # 数据收集器
    compute_metrics=compute_metrics  # 评估指标计算函数
)

# 开始训练
trainer.train()

# 正确保存模型到指定目录
trainer.save_model('./final_model')

# 计算并打印执行时间
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Code ran in {elapsed_time:.4f} seconds")
