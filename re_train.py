import re

#正则表达式训练

text = "111apple123banana456cherry"

# 编译正则表达式，生成Pattern对象
pattern = re.compile(r'\d+', re.IGNORECASE)
# 在字符串中搜索正则表达式匹配到的第一个位置的值，并返回匹配到的对象
result = re.search(pattern, text)
print("search text:", result)

# 搜索字符串，并以列表形式返回匹配到的全部字符串
result = re.findall(pattern, text)
print("findall text:", result)

# 在目标字符串开始位置匹配正则表达式，并返回Match对象，若未匹配成功则返回None
result = re.match(pattern, text)
print("match text:", result)

# split()方法将一个字符串按照正则表达式匹配的结果进行分割，并以列表形式返回数据。
# 如果正则表达式匹配到的字符恰好在字符串开头或者结尾，
# 则返回的分割后的字符串列表首尾元素都为空项，此时需要手动去除空项
result = re.split(pattern, text)
print("Split text:", result)

# sub()方法将正则表达式匹配到的字符串替换为指定的字符串，并返回替换后的字符串
result = re.sub(pattern, "*", text)
print("sub text:", result)
