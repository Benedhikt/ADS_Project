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
import joblib


# Set the title of the app
st.title("🏡 Boston Housing Price Prediction")
st.write("This app predicts house prices based on various features.")

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
    crim = st.sidebar.slider("CRIM", 0.0, 10.0, 1.0, step=0.1)
    zn = st.sidebar.slider("ZN", 0, 100, 20)
    indus = st.sidebar.slider("INDUS", 0.0, 30.0, 5.0, step=0.1)
    chas = st.sidebar.slider("CHAS", 0, 1, 0)
    nox = st.sidebar.slider("NOX", 0.0, 1.0, 0.5, step=0.01)
    rm = st.sidebar.slider("RM", 3.0, 10.0, 6.0, step=0.1)
    age = st.sidebar.slider("AGE", 0, 100, 50)
    dis = st.sidebar.slider("DIS", 0.0, 15.0, 5.0, step=0.1)
    rad = st.sidebar.slider("RAD", 1, 24, 5)
    tax = st.sidebar.slider("TAX", 100, 800, 300)
    ptratio = st.sidebar.slider("PTRATIO", 10.0, 30.0, 15.0, step=0.1)
    b = st.sidebar.slider("B", 0.0, 400.0, 200.0, step=0.1)
    lstat = st.sidebar.slider("LSTAT", 0.0, 40.0, 10.0, step=0.1)

    # Store inputs in a DataFrame
    data = {
        "crim": crim, "zn": zn, "indus": indus, "chas": chas, "nox": nox, "rm": rm,
        "age": age, "dis": dis, "rad": rad, "tax": tax, "ptratio": ptratio, "b": b, "lstat": lstat
    }
    
    return pd.DataFrame([data])

user_input_data = user_input()


## Loading the trained model
LR = joblib.load("linear_regression_model.pkl")

# Prediction
prediction = LR.predict(user_input_data)

st.write(f"### 💰 Predicted House Price: **${prediction[0]*1000:.2f}**")
##st.write(f"Predicted House Price: ${prediction[0]*1000:.2f}")

# Plotting
st.subheader("📊 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)  # Pass figure explicitly

# Histogram of CRIM
st.write("### Histogram Distribution of CRIM")
st.write("Here we explore the distributions of CRIM in the dataset.")

st.write("Distribution of CRIM:")
fig, ax = plt.subplots()
ax.hist(df['crim'], bins=20, color='skyblue', edgecolor='black')
ax.set_title("CRIM Distribution")
ax.set_xlabel("CRIM")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Scatter Plot of CRIM vs ZN
import plotly.express as px

st.write("### Scatter Plot of CRIM vs ZN")
st.write("Here we explore the relationship between CRIM and ZN in the dataset.")

st.subheader("📈 House Prices vs Rooms (RM)")
fig = px.scatter(df, x="rm", y="medv", color="lstat",
                 title="House Prices vs Number of Rooms",
                 labels={"rm": "Average Number of Rooms", "medv": "Median Value of House ($1000s)"})
st.plotly_chart(fig)


st.write("### Histogram of House Prices")
st.subheader("📊 Distribution of House Prices")
fig = px.histogram(df, x="medv", nbins=30, title="House Price Distribution")
st.plotly_chart(fig)



# Slider for feature selection
feature = st.sidebar.selectbox("Select a feature to plot", df.columns)

# Plot selected feature
st.write(f"Distribution of {feature}:")
fig, ax = plt.subplots()
ax.hist(df[feature], bins=20, color='skyblue', edgecolor='black')
ax.set_title(f"{feature} Distribution")
ax.set_xlabel(feature)
ax.set_ylabel("Frequency")
st.pyplot(fig)


st.download_button(
    label="📥 Download Prediction",
    data=str(prediction[0]),
    file_name="prediction.txt",
    mime="text/plain"
)
