import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

# Set the title of the app
st.title("Boston Housing Price Prediction")

# Importing dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

# Show data preview
st.write("Dataset Preview:")
st.write(df.head())

# Split data
X = df.drop(columns=['medv'])  # Features
y = df['medv']  # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=142)

# Train a Linear Regression model
LR = LinearRegression()
LR.fit(X_train, y_train)

# Model evaluation
y_pred = LR.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Display RMSE and R^2 Score
st.write(f"Model Evaluation:")
st.write(f"RMSE: {rmse}")
st.write(f"R^2 Score: {LR.score(X_test, y_test)}")

# User Input
st.sidebar.header("User Input Parameters")
def user_input():
    lstat = st.sidebar.slider('LSTAT', min_value=0, max_value=40, value=10)
    indus = st.sidebar.slider('INDUS', min_value=0, max_value=40, value=10)
    nox = st.sidebar.slider('NOX', min_value=0, max_value=1, value=0.5)
    rm = st.sidebar.slider('RM', min_value=3, max_value=10, value=6)
    age = st.sidebar.slider('AGE', min_value=0, max_value=100, value=30)
    dis = st.sidebar.slider('DIS', min_value=0, max_value=12, value=6)
    rad = st.sidebar.slider('RAD', min_value=1, max_value=24, value=5)
    tax = st.sidebar.slider('TAX', min_value=100, max_value=700, value=300)
    ptratio = st.sidebar.slider('PTRATIO', min_value=10, max_value=30, value=20)
    b = st.sidebar.slider('B', min_value=0, max_value=400, value=300)
    lstat = st.sidebar.slider('LSTAT', min_value=0, max_value=40, value=10)
    
    data = {'lstat': lstat, 'indus': indus, 'nox': nox, 'rm': rm, 'age': age,
            'dis': dis, 'rad': rad, 'tax': tax, 'ptratio': ptratio, 'b': b}
    features = pd.DataFrame(data, index=[0])
    return features

user_input_data = user_input()

# Prediction
prediction = LR.predict(user_input_data)

st.write(f"Predicted House Price: ${prediction[0]*1000:.2f}")

# Plotting
st.write("Correlation Heatmap:")
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
st.pyplot()

