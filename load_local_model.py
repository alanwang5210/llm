from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import pipeline

model_name_or_path = "D:\\workspace\\alan\\pyRag\\models\\BAAI\\bge-m3"
tokenizer = AutoTokenizer.from_pretrained("D:\\workspace\\alan\\pyRag\\models\\BAAI\\bge-m3")
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
result = classifier("今天心情很好")
print(result)
