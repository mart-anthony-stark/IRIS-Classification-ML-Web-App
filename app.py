import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.primaryColor="#000000"
st.backgroundColor="#FFFFFF"
st.secondaryBackgroundColor="#F0F2F6"
st.textColor="#262730"
st.font="sans serif"

st.write("""


![IRIS FLOWER](https://prutor.ai/wp-content/uploads/iris-flower-classification.jpg)
# IRIS FLOWER CLASSIFICATION

  Iris flower classification is a very popular machine learning project. The iris dataset contains three 
classes of flowers, Versicolor, Setosa, Virginica, and each class contains 4 features, ‘Sepal length’, 
‘Sepal width’, ‘Petal length’, ‘Petal width’. The aim of the iris flower classification is to predict 
flowers based on their specific features.
""")


st.sidebar.header('Predict Input Parameters')
st.sidebar.subheader('User Input parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()