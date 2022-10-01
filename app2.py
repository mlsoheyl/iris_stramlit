# ----------------------------------------------
# Initialization
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# ----------------------------------------------
# create streamlit project
st.write('Simple web app for predicting IRIS Flower')


# ----------------------------------------------
# input data for predict


def user_input_features():
     sepal_length = st.sidebar.slider('Sepal Lenght', 0.2, 5.1)
     sepal_width = st.sidebar.slider('Sepal width', 0.2, 4.9)
     petal_length = st.sidebar.slider('petal length', 0.2, 4.7)
     petal_width = st.sidebar.slider('petal width', 0.2, 4.6)
     data = {
          'sepal_length': sepal_length,
          'sepal_width': sepal_width,
          'petal_length': petal_length,
          'petal_width': petal_width
          }
     features = pd.DataFrame(data, index=[0])
     return features


df = user_input_features()

# ----------------------------------------------
# load Iris
iris = load_iris()
x = iris.data
y = iris.target

# ----------------------------------------------
# modeling
clf = RandomForestClassifier()
clf.fit(x, y)

# ----------------------------------------------
# prediction
y_pre = clf.predict(df)

# st.write(y_pre)
st.subheader('The prediction is: ')
st.write(iris.target_names[y_pre])
st.balloons()

st.write(iris.target_names[0:3])
