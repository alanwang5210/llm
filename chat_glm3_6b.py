import os
import warnings

import torch

warnings.filterwarnings('ignore')
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

model_name_or_path = 'E:\\ProgramData\\model\\ZhipuAI\\chatglm3-6b'
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     llm_int8_threshold=6.0,
#     llm_int8_has_fp16_weight=False,
# )
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, trust_remote_code=True)

dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("1111111111111")
model = AutoModel.from_pretrained(model_name_or_path,
                                  return_dict=True, trust_remote_code=True, use_safetensors=False,
                                  local_files_only=True, torch_dtype=dtype)
print("222222222222")
f = open('读者的本地数据集文件地址', encoding='UTF-8')
results = f.readlines()
# prompt和response分离
lines = []
data_prompt = []
data_response = []
print(len(results))
for i in range(0, len(results)):
    if results[i] == '\n':
        continue
    else:
        lines.append(results[i])
for i in range(0, len(lines)):
    if i % 2 == 0:
        context = '你现在是一名小学数学老师，需要对学生提出的问题进行回答。有一位同学向你询问下述内容。'
        data_prompt.append(context + lines[i])
    else:
        data_response.append(lines[i])
