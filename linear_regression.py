import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.autograd import Variable


# 构建线性回归模型
# 定义线性回归模型类
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        # 创建一个线性层，输入维度1，输出维度1
        self.linear = nn.Linear(1, 1)

    # 前向传播函数，定义模型的计算过程
    def forward(self, x):
        out = self.linear(x)  # 使用线性层进行计算
        return out


# 创建LinearRegression类的实例（即线性回归模型）
model = LinearRegression()

# 打印模型的结构，查看模型的层次和参数
print(model)

# 设置训练过程中的一些超参数\
num_epochs = 1000  # 训练的总轮数
learning_rate = 1e-2  # 学习率，控制每次参数更新的步长
loss_fn = nn.MSELoss()  # 使用均方误差作为损失函数
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # 使用随机梯度下降（SGD）优化器

# 生成模拟的训练数据：输入 x 从 -1 到 1 的 100 个等间距点
x = Variable(torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1))  # 将 1D 张量转为 2D 张量
# 生成输出 y，y = 2x + 0.2 + 随机噪声
y = Variable(x * 2 + 0.2 + torch.rand(x.size()))  # 模拟带有噪声的线性关系

for epoch in range(num_epochs):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print("[{}/{}] loss:{:.4f}".format(epoch + 1, num_epochs, loss))

plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=5)
plt.text(0.5, 0, 'Loss = %.4f' % loss.data.item(), fontdict={'size': 20, 'color': 'red'})
plt.show()

[w, b] = model.parameters()
print(w, b)
