import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# IRIS FLOWER CLASSIFICATION

  Iris flower classification is a very popular machine learning project. The iris dataset contains three 
classes of flowers, Versicolor, Setosa, Virginica, and each class contains 4 features, ‘Sepal length’, 
‘Sepal width’, ‘Petal length’, ‘Petal width’. The aim of the iris flower classification is to predict 
flowers based on their specific features.
""")