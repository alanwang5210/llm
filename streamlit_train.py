import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier


# 用户输入数据
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length, 'sepal_width': sepal_width,
            'petal_length': petal_length, 'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()
# 加载Iris数据集并训练模型
iris = datasets.load_iris()
X = iris.data
Y = iris.target
clf = RandomForestClassifier()
clf.fit(X, Y)
# 对输入数据进行分类并展示
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)
st.write('分类结果：' + iris.target_names[prediction][0])



df = pd.DataFrame({'col1': [1,2,3]})
