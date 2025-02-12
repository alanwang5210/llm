import sentencepiece as spm

# 配置参数
# 训练数据文件路径, 多个文件用逗号分隔, 同时支持读取压缩文件
input_file = "data/data.txt"
# 模型输出文件的前缀
model_prefix = "./data/spm_model"
# 词汇表大小（如 8000、16000）。
vocab_size = 150
# --character_coverage：字符覆盖率，默认为 1.0，适用于英文。如果是中文或其他语言，可设为 0.995。
# --model_type：模型类型，有以下选项：
#     unigram（默认）：基于概率的子词模型。
#     bpe：字节对编码。
#     word：词级别模型。
#     char：字符级别模型。

# 训练 SentencePiece 模型
spm.SentencePieceTrainer.Train(
    f"--input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size} --character_coverage=1.0 --model_type=bpe"
)

# SentencePiece 的多次训练不会自动合并结果，因为每次训练会基于提供的输入数据重新生成一个独立的分词模型和词汇表。
# 模型训练的结果是自包含的，即它不依赖于之前的训练结果，也不会自动更新已有模型。
# 可以进行后期词汇表的合并结果
# 读取多个 vocab 文件
# vocab_files = ["model1.vocab", "model2.vocab"]
# combined_vocab = Counter()
#
# for vocab_file in vocab_files:
#     with open(vocab_file, "r", encoding="utf-8") as f:
#         for line in f:
#             token, score = line.strip().split("\t")
#             combined_vocab[token] += float(score)
#
# # 保存合并后的 vocab
# with open("combined.vocab", "w", encoding="utf-8") as f:
#     for token, score in combined_vocab.most_common():
#         f.write(f"{token}\t{score}\n")

print(f"Model trained. Files saved as {model_prefix}.model and {model_prefix}.vocab")
