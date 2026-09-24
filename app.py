import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="ML Lifecycle App - Vishnu", layout="centered")

st.title("ML Lifecycle App - By Vishnu")
st.write("Complete ML Lifecycle Demo")

# 1. Data Ingestion & Preparation
st.header("1. Data Ingestion & Preparation")
data = {'Size_sqft': [1000,1500,1200,1800,2000,1100,1600,1300], 'Price': [50,80,65,95,110,55,85,70]}
df = pd.DataFrame(data)
st.dataframe(df)

# 2. Model Training
st.header("2. Model Training")
X = df[['Size_sqft']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
accuracy = r2_score(y_test, model.predict(X_test))
st.write(f"Model: Linear Regression | Accuracy: {accuracy:.2f}")

# 3. Deployment
st.header("3. Deployment - Live Prediction")
size = st.slider("Select Size sqft", 500, 3000, 1500)
if st.button("Predict Price"):
    pred = model.predict([[size]])
    st.success(f"Predicted Price: {pred[0]:.2f} Lakh")

# 4. Management
st.header("4. Model Management")
st.write("Monitoring accuracy, versioning, future retraining")

st.info("Project by Vishnu")