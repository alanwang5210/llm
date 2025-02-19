import os

from peft import get_peft_model, AdaLoraConfig
from transformers import AutoModelForCausalLM

# 设置HF的镜像地址（可选）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Prefix tuning 使用Prefix tuning时需要设置参数指令文本的长度(num_virtual_tokens)。这个参数一般设置为10～20。
# model_name_or_path = "princeton-nlp/unsup-simcse-roberta-base"
# peft_type = PeftType.PREFIX_TUNING
# peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=20)
# lr = 1e-2
# model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)

# Prompt tuning 使用Prompt tuning时需要设置参数指令文本的长度(num_virtual_tokens)
# model_name_or_path = "princeton-nlp/unsup-simcse-roberta-base"
# peft_type = PeftType.PROMPT_TUNING
# peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=20)
# lr = 1e-3
# model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path,
#                                                            return_dict=True)


# P-tuning v1，使用P-tuning v1时需要设置两个参数：一个是MLP中间层的参数(encoder_hidden_size)，
# 另一个是指令文本的长度(num_virtual_tokens)。
# model_name_or_path = "princeton-nlp/unsup-simcse-roberta-base"
# peft_type = PeftType.P_TUNING
# peft_config = PromptEncoderConfig(task_type="SEQ_CLS",
#                                   num_virtual_tokens=20,
#                                   encoder_hidden_size=128)
# lr = 1e-3
# model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)

# P-tuning v2
# model_name_or_path = "princeton-nlp/unsup-simcse-roberta-base"
# peft_config = PrefixTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=30)
# model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

# LoRA LoRA微调最为关键的步骤是秩r的选取，它决定了低秩矩阵的大小。
# model_name_or_path = "princeton-nlp/unsup-simcse-roberta-base"
# peft_type = PeftType.LORA
# peft_config = LoraConfig(task_type="SEQ_CLS",
#                          inference_mode=False,
#                          r=8,
#                          lora_alpha=16,
#                          lora_dropout=0.1)
# lr = 3e-4
# model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)

# IA3
# model_name_or_path = "princeton-nlp/unsup-simcse-roberta-base"
# peft_config = IA3Config(task_type="CAUSAL_LM",
#                         target_modules=["key", "value", "output.dense"],
#                         inference_mode=False,
#                         feedforward_modules=["output.dense"])
# model = AutoModelForCausalLM.from_pretrained(model_name_or_path)


# AdaLoRA
model_name_or_path = "princeton-nlp/unsup-simcse-roberta-base"
peft_config = AdaLoraConfig(peft_type="ADALORA",
                            task_type="SEQ_2_SEQ_LM",
                            r=8,
                            lora_alpha=32,
                            target_modules=["key", "value"],
                            lora_dropout=0.01)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, return_dict=True)

# 获取PEFT模型
peft_model = get_peft_model(model, peft_config)

# 打印可以训练的参数
peft_model.print_trainable_parameters()

# 保存微调后的模型
peft_model.save_pretrained("./model")

# 登录 Hugging Face
# notebook_login()

# 将微调模型上传到 Hugging Face Hub
# peft_model.push_to_hub("my_peft_model")
