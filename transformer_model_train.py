import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## 数据准备与参数设置

# 训练数据包含3个层次——编码器输入、解码器输入和解码器输出。
# 其中，解码器输入需要以“S”标识句子的开始，解码器输出则需要以“E”标识句子的结束，
# 而“P”代表占位符（由于sentences中的第1句和第3句中文语句比第2句中文语句短，因此需要占位符P进行补充

# src_vocab表示词源字典，其中每个字符对应一个索引值。
# 随后，将src_vocab转换为字典数据类型，并保存其长度为src_vocab_size。
sentences = [['我 是 教 师 P', 'S I am a teacher', 'I am a teacher E'],
             ['我 喜 欢 教 学', 'S I like teaching P', 'I like teaching P E'],
             ['我 是 厨 师 P', 'S I am a cook', 'I am a cook E']]
src_vocab = {'P': 0, '我': 1, '是': 2, '教': 3, '师': 4, '喜': 5, '欢': 6, '学': 7, '厨': 8}
src_idx2word = {src_vocab[key]: key for key in src_vocab}
src_vocab_size = len(src_vocab)

# 下面处理目标词表(tgt_vocab)。与上面的方法类似，
# 每个字母或单词对应固定的索引值，转换为字典数据类型并保存其长度。
tgt_vocab = {'P': 0, 'S': 1, 'E': 2, 'I': 3, 'am': 4, 'a': 5, 'teacher': 6, 'like': 7, 'teaching': 8, 'cook': 9}
idx2word = {tgt_vocab[key]: key for key in tgt_vocab}
tgt_vocab_size = len(tgt_vocab)

# 设置中文句子固定最大长度(src_len)和英文句子固定最大长度(tgt_len)
src_len = len(sentences[0][0].split(" "))
tgt_len = len(sentences[0][1].split(" "))


# 定义make_data()方法，将sentences转化为字典索引，也就是将sentences的每个句子转化为数字向量
# 例如，句子“我是教师P”经过make_data()方法处理后将转变为“[1, 2,3, 4, 0]。
def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]
        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


enc_inputs, dec_inputs, dec_outputs = make_data(sentences)


# 继承torch.utils.data包的Data类以定义新的MyDataSet类，用于加载训练数据
class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


# 实例化DataLoader对象，并将数据转化为批大小为2的分组数据。
loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)

# 对编码器、解码器的个数、前向传播隐层的维度等参数进行设置。
d_model = 512  # 嵌入的维度
d_ff = 2048  # 前向传播隐层维度
d_k = d_v = 64  # K、V矩阵的维度
n_layers = 6  # 编码器和解码器的数量
n_heads = 8  # 多头自注意力数


##位置编码
# 在Transformer中，由于输入文字经过向量化后成为字向量，因此每个句子可以用矩阵表示。
# 而编码器输入由字向量与位置信息（由公式计算得出）的加和得到，以便在并行计算时得到字之间的顺序关系。
# pos_table代表位置信息矩阵，它与输入矩阵enc_inputs相加后可以得到带有位置信息的字向量
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array([
            [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
            if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])
        self.pos_table = torch.FloatTensor(pos_table).to(device)

    def forward(self, enc_inputs):
        enc_inputs += self.pos_table[:enc_inputs.size(1), :]
        return self.dropout(enc_inputs.to(device))


##掩码操作
# 由于输入中包含“P”这样的占位符，占位符对于句子没有实际含义，
# 因此可以将其掩码。如下代码将定义get_attn_pad_mask()方法，用于掩码占位符。
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


# 在解码器部分，需要掩码输入信息。例如，对于句子“S I ama teacher”，
# 首先，会将“S”后的字符进行掩码，并由解码器预测出第一个输出“I”。
# 随后，将“S”和“I”输入解码器中，得到下一个预测结果“am”，
# 依此类推。所以，我们需要将解码器的输入矩阵转化为上三角矩阵，对句子中的每个被预测对象及其后续单词进行掩码。
def get_attn_subsequence_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask


##注意力计算
# 在多头注意力机制（之前已经设置注意力头数为8）中，矩阵被拆分为8个小型矩阵。
# 此处通过如下代码定义缩放点积注意力(ScaledDotProductAttention)类和多头注意力类(MultiHeadAttention)
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        return nn.LayerNorm(d_model).to(device)(output + residual), attn


##前馈神经网络
# 首先将输入经过两个全连接层的计算后得到结果(output)，然后将计算结果与输入(residual)进行相加，并进行归一化。
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).to(device)(output + residual)


## 编码器与解码器
# 由前面的内容可知，整个Transformer包含多组解码器和编码器。此处先定义单层的编码器和解码器
# （按照Transformer的架构实例化各个部分，如多头自注意力机制、前馈神经网络等）。
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


# 在定义单层解码器类时，初始化方法包含两个多头自注意力模块。
# 第一个多头自注意力模块的输入矩阵Q、K、V的值与解码器的输入相等。
# 第二个多头自注意力模块的矩阵Q的值来自解码器，矩阵K、V的值来自编码器的输出。
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


# 在定义完整的编码器结构时，首先将输入转化为512维的字向量，并且在字向量中加入位置信息。
# 随后，对句子中的占位符进行掩码，然后将其输入6层的编码器模块中，
# 上一层的输出可作为下一层的输入，如此循环计算，得到最终结果。
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


# 解码器的结构与上述编码器的结构类似，不同的是，首先将英文单词进行索引，并转化为512维的字向量，
# 随后在字向量中加入位置信息，并掩码句子中的占位符，然后通过6层解码器结构进行计算。
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs).to(device)
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).to(device)
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).to(device)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0).to(device)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


##构建Transformer
# 这里对前面介绍的组件进行组合（编码器和解码器），以构建Transformer类。
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.Encoder = Encoder().to(device)
        self.Decoder = Decoder().to(device)
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).to(device)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attns = self.Encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.Decoder(
            dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


##模型训练
# 首先实例化Transformer类，并定义优化器（随机梯度下降）和损失函数（交叉熵）。
model = Transformer().to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

# 然后构建训练循环。此处将前面生成的loader作为输入数据，共进行50个轮次的训练。
for epoch in range(50):
    for enc_inputs, dec_inputs, dec_outputs in loader:
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, dec_outputs.view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# 接下来对训练完成的模型进行测试，其中，test()方法用于计算解码器层的输入
def test(model, enc_input, start_symbol):
    enc_outputs, enc_self_attns = model.Encoder(enc_input)
    dec_input = torch.zeros(1, tgt_len).type_as(enc_input.data)
    next_symbol = start_symbol
    for i in range(0, tgt_len):
        dec_input[0][i] = next_symbol
        dec_outputs, _, _ = model.Decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
    return dec_input


enc_inputs, _, _ = next(iter(loader))
predict_dec_input = test(model, enc_inputs[0].view(1, -1).to(device), start_symbol=tgt_vocab["S"])
predict, _, _, _ = model(enc_inputs[0].view(1, -1).to(device), predict_dec_input)
predict = predict.data.max(1, keepdim=True)[1]
print([src_idx2word[int(i)] for i in enc_inputs[0]], '->', [idx2word[n.item()] for n in predict.squeeze()])
