import numpy as np
import torch

# 用于声明一个未初始化的张量
# x = torch.empty(5, 3)

# torch.rand()方法用于随机初始化一个张量
# x = torch.rand(5, 3)

# 直接向torch.tensor()方法传递具体数值列表
# x = torch.tensor([6.6, 9.9])


# torch.zeros()方法用于创建数值均为0的张量，
# torch.ones()可以创建数值均为1的张量
# zeros = torch.ones(5, 3, dtype=torch.long)


# torch.LongTensor()方法用于创造一个统一的长张量
# x = torch.LongTensor(3, 4)

# torch.FloatTensor()方法可以创建浮点数类型的张量
# x = torch.FloatTensor(3, 4)

# torch.arange()方法可以创建由10以内数值组成的张量，并使用torch.view()方法将该张量的形状变为2×5
# x = torch.arange(10, dtype=torch.float)
# print(x)
# torch.view()方法用于重新塑形张量，即改变张量的形状而不改变其数据
# print(x.view(2, 5))

# torch.permute()方法可用于改变张量的维度顺序。它返回一个新的张量，其维度按照指定的顺序重新排列
# x1 = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
# print("x1: \n", x1)
# print("\nx1.shape: \n", x1.shape)
# print("\nx1.view(3, -1): \n", x1.view(3, -1))
# print("\nx1.permute(1, 0): \n", x1.permute(1, 0))

# 张量的加法有3种实现方式，包括直接使用“+”运算符、torch.add()方法以及直接修改Tensor变量方法
# 方法3使用了直接修改Tensor变量的方法，会直接修改原变量的值，这类方法统一在方法名中带有后缀“_”
# x = torch.rand(5, 3)
# y = torch.rand(5, 3)
# print('方法1:', x + y)
# result = torch.empty(5, 3)
# torch.add(x, y, out=result)
# print('方法2:', result)
# y.add_(x)
# print('方法3:', y)


# NumPy持大型数组、多维数组和矩阵。开发者可以通过tensor. numpy()方法将PyTorch张量转换为NumPy数组，
# PyTorch张量和NumPy数组将共享底层内存位置，改变任何一个将同时影响\另一个
# a = torch.ones(5)
# print(a)
# b = a.numpy()
# print(b)

# 通过torch.from_numpy()方法将NumPy数组转换为PyTorch张量
# a = np.ones(5)
# b = torch.from_numpy(a)
# print(b)


# Autograd是PyTorch的重要工具之一，具有对张量进行自动微分的功能。
x = torch.zeros(3, 3, requires_grad=True)
print(x)
# 对张量x执行加法计算操作，并输出其操作结果y的grad_fn属性
y = x + 2
print(y)
print(y.grad_fn)

# 乘法
m = y * y * 3
# 取平均值
n = m.mean()
print(m)
print(n)

# 调用backward()方法进行反向传播，通过计算图自动计算梯度
n.backward()
print(x.grad)

# 计算对数函数的梯度
x = torch.tensor([0.5, 0.75], requires_grad=True)
y = torch.log(x[0] * x[1])
y.backward()
print(x.grad)