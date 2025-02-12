from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, DataCollatorWithPadding

data_files = {"train": "review.csv", "test": " review.csv "}
raw_datasets = load_dataset("Chinese-Vicuna/guanaco_belle_merge_v1.0")


model_name_or_path = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path,num_labels=5)
def tokenize_function(examples):
    return tokenizer(examples["text"],padding='max_length',truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function,batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print(tokenized_datasets)

