import sentencepiece as spm

sp = spm.SentencePieceProcessor()

# 加载训练好的模型
sp.Load("spm_model.model")

# 分词
text = "SentencePiece is awesome!"
# 使用模型进行分词
# SentencePiece 支持以下几种类型的输出：
# 字符串（str）：输出子词的字符串形式。
# 整数（int）：输出子词的索引（数字形式）。
# 字典形式（dict）：输出一个包含子词和其对应词频的字典。
tokens = sp.Encode(text, out_type=str)
# encode_as_pieces()方法用于对句子进行分词，
# encode_as_ids()方法则可以将句子转化为ID序列（因为词表的每个单词均有ID值

# ▁ 是 SentencePiece 中的分隔符，表示子词的边界。▁ 是 SentencePiece 用来表示词与词之间的空格。
print("Tokens:", tokens)

#  使用模型进行解码
decoded_text = sp.Decode(tokens)

# 通过decode_pieces()方法和decode_ids()方法将分词结果进行还原
print("Decoded:", decoded_text)
