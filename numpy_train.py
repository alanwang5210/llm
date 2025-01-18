import numpy as np

# 通过列表生成一维数组，并输出数据类型
data = [1, 2, 3, 4, 5]
x = np.array(data)
print(x)
print(x.dtype)

# 通过zeros()方法创建一个长度为5、元素均为0的一维数组
print(np.zeros(5))

# 通过zeros()方法创建一个一维长度为2、二维长度为3的二维零数组
print(np.zeros((2, 3)))

# 通过ones()方法创建一个一维长度为2、二维长度为3、元素均为1的二维数组
print(np.ones((2, 3), dtype=np.int16))

# 通过arange方法生成连续元素
print(np.arange(5, 10, dtype=np.int16))

# 矢量运算是指把大小相同的数组间的运算应用在数组的各个元素上
x = np.array([1, 2, 3, 4, 5])
print(x * 2)
print(x > 2)
y = np.array([2, 3, 4, 5, 6])
print(x + y)
print(x > y)

# 获取转置（矩阵）数组
k = np.arange(8)
m = k.reshape(2, 2, 2)
print(m)

# 获取数组及其转置数组的乘积
# print(np.dot(m, m.T))


# print(m.transpose(np.arange(8), axes=1))