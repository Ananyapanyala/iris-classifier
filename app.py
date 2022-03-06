import streamlit as st
st.title("IRIS classifier-API")
sl = st.slider('sepal lenght',4.3,7.9,0.5)
sw = st.slider('sepal width',2.0,4.4,0.5)
pl = st.slider('petal lenght',1.0,6.9,0.5)
pw = st.slider('petal width',0.1,2.5,0.5)
from sklearn.datasets import load_iris
iris = load_iris()
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(iris.data,iris.target)
op = model.predict([[sl,sw,pl,pw]])
op = iris.target_names[op[0]]
st.title(op)
                                                                                                  
