import os
from peft import LoraConfig
from peft import get_peft_model
from transformers import AutoModelForMaskedLM, AutoTokenizer
from huggingface_hub import notebook_login

# 设置HF的镜像地址（可选）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 配置LoRA方法
peft_config = LoraConfig(
    r=8,  # 低秩矩阵的秩（决定参数量，默认8）
    lora_alpha=32,  # 缩放系数（影响适配器权重对原模型的贡献）
    target_modules=["encoder.layer.*.attention.self.query",  # 需要根据模型结构调整
                    "encoder.layer.*.attention.self.value"],  # LoRA的目标模块
    lora_dropout=0.1,  # 防止过拟合的丢弃率
    bias="none",  # 是否训练偏置项（可选"none"/"all"/"lora_only"）
    task_type="MASKED_LM"  # 使用掩蔽语言模型任务类型（适合BERT等模型）
)

# 加载模型和Tokenizer
model_name_or_path = "bert-base-chinese"  # 使用 BERT 作为示例
tokenizer_name_or_path = "bert-base-chinese"

model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

# 获取PEFT模型
peft_model = get_peft_model(model, peft_config)

# 打印可以训练的参数
peft_model.print_trainable_parameters()

# 保存微调后的模型
peft_model.save_pretrained("./model")

# 登录 Hugging Face
notebook_login()

# 将微调模型上传到 Hugging Face Hub
peft_model.push_to_hub("my_peft_model")