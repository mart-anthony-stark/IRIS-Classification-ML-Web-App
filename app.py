import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
![IRIS FLOWER](https://prutor.ai/wp-content/uploads/iris-flower-classification.jpg)
# IRIS FLOWER CLASSIFICATION
  The iris dataset contains three classes of flowers, Versicolor, Setosa, Virginica, and each class contains 4 features, ‘Sepal length’, 
‘Sepal width’, ‘Petal length’, ‘Petal width’.
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

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Flower Classification labels and Index')
st.write("""
| Setosa: 0 | Versicolor: 1 | Virginica: 2 |
|-----------|---------------|--------------|
""")

st.write("""

### Predicted output: %(prediction)s
Probability:
| Setosa: 0 | Versicolor: 1 | Virginica: 2 |
|-----------|---------------|--------------|
|  %(p1)s   |     %(p2)s    |     %(p3)s   |
""" % {
  "prediction": iris.target_names[prediction][0].upper(),
  "p1": prediction_proba[0][0],
  "p2": prediction_proba[0][1],
  "p3": prediction_proba[0][2]
})