import pandas as pd
from pandas import DataFrame
from pandas import Series

import numpy as np

# x = pd.Series(['a', 'b', 'c', 'd'], [1, 2, 3, 4])
# x = pd.Series({"a": 1, "b": 2, "c": 3})
# y = pd.Series(np.arange(5), np.arange(9, 4, -1))

# x = pd.Series(['Jack', 'Tony', 'Jim'], ['1', '2', '3'])
# x["4"] = 'Danny'  # 添加
# x = x[1:4]  # 切片
# x.drop(labels='2', inplace=True)  # 删除
# x.loc['4'] = 'Lucy'  # 修改

# x = pd.Series([4, 7, 3, 2], ['b', 'a', 'd', 'c'])
# y = x.sort_index(ascending=False)
# y = x.reindex(['a', 'b', 'c', 'd', 'e'], fill_value=0)

df = DataFrame({'name': Series(['Ken', 'Kate', 'Jack']), 'age': [21, 18, 15], 'sex': [21, 18, 20]})

print(df.at[2, 'age'])

df.to_csv('hello.csv')