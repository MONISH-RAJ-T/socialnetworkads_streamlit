      
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import json
from streamlit_lottie import st_lottie
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

st.title("Simple Streamlit ML Model")

# Fix 1: Handle Lottie file loading safely
# def load_lottiefile(filepath: str):
#     try:
#         with open(filepath, "r", encoding="utf-8") as f:
#             return json.load(f)
#     except Exception as e:
#         st.error(f"Error loading Lottie file: {e}")
#         return None

# with st.sidebar:
#     st.title("List of Contents")
    
#     # Fix 2: Use relative path and ensure file exists
#     lottie_coding = load_lottiefile("M:\Projects\streamlitproject-main\Animation - 1712585147924.json")  # Renamed to .json if needed
    
#     if lottie_coding:
#         st_lottie(lottie_coding, 
#                  speed=1,
#                  reverse=False,
#                  loop=True,
#                  quality="low",
#                  height=None,
#                  width=None,
#                  key=None)
with st.sidebar:
    st.title("List of Contents")   
    choice = st.radio("Menu", ["Home", "Train Model", "Predict"])

if choice == "Train Model":
    st.header("Train a Machine Learning Model")

    # Fix 3: Use uploaded file instead of hardcoded path
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("File uploaded successfully.")
            
            # Display sample data
            st.write("Sample data:")
            st.write(df.head())

            # Fix 4: Proper label encoding
            label_encoder = LabelEncoder()
            for column in df.columns:
                if df[column].dtype == 'object':
                    df[column] = label_encoder.fit_transform(df[column])

            # Select target column
            target_column = st.selectbox("Select target column", df.columns)

            # Correlation heatmap
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)

            # Split data
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Model training
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            # Evaluation
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Model Accuracy: {accuracy:.2f}")

            # Feature importance
            st.header("Feature Importance")
            feature_importance = pd.Series(model.feature_importances_, index=X.columns)
            st.bar_chart(feature_importance.sort_values(ascending=False))

            # Save model
            with open("trained_model.pkl", "wb") as f:
                pickle.dump(model, f)
                
        except Exception as e:
            st.error(f"Error processing file: {e}")

elif choice == "Predict":
    st.header("Make Predictions")
    
    try:
        # Fix 5: Load model safely
        with open("trained_model.pkl", "rb") as model_file:
            model = pickle.load(model_file)
            
        # Fix 6: Create proper input form
        with st.form("prediction_form"):
            st.write("Enter feature values:")
            
            UserID = st.number_input("User ID", min_value=0)
            Gender = st.selectbox("Gender", ["Male", "Female"])
            Age = st.number_input("Age", min_value=18, max_value=100)
            EstimatedSalary = st.number_input("Estimated Salary", min_value=0)
            
            submitted = st.form_submit_button("Predict")
            
            if submitted:
                # Convert gender to numerical
                gender_mapping = {"Male": 1, "Female": 0}
                gender_num = gender_mapping[Gender]
                
                # Create feature array
                features = np.array([[UserID, gender_num, Age, EstimatedSalary]])
                
                # Make prediction
                try:
                    prediction = model.predict(features)
                    st.success(f"Predicted Purchase: {'Yes' if prediction[0] else 'No'}")
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    
    except FileNotFoundError:
        st.error("Model not found. Please train a model first.")
    except Exception as e:
        st.error(f"Error loading model: {e}")

if choice == "Home":
    st.header("Welcome to the ML App")
    st.write("Select an option from the sidebar to get started!")

