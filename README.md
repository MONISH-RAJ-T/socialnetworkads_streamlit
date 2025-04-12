# README - Simple Streamlit ML Model

## Overview

This is a simple Streamlit web application that allows users to train a Machine Learning model and make predictions. The model used is a Random Forest Classifier trained on the `Social_Network_Ads.csv` dataset.

## Features

- **Home Page**: Displays a sidebar with an animated Lottie file.
- **Train Model**: Allows users to upload a CSV file, preprocess data, train a model, and view feature importance.
- **Predict**: Loads a pre-trained model and allows users to input values for prediction.

## Installation

Ensure you have Python installed along with the following dependencies:

```bash
pip install streamlit pandas seaborn matplotlib scikit-learn numpy streamlit-lottie
```

## Usage

1. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```
2. **Navigate the Interface:**
   - Use the sidebar to select between `Home`, `Train Model`, and `Predict`.
   - Upload the `Social_Network_Ads.csv` file to train the model.
   - Input values under `Predict` to use the pre-trained model.

## File Structure

- `app.py` - Main Streamlit application.
- `Social_Network_Ads.csv` - Sample dataset.
- `trained_model2.pkl` - Pre-trained model file.
- `Animation.json` - Lottie animation file.

## Download

[Click here to download the project files](https://github.com/MONISH-RAJ-T/streamlitproject.git)

## Author

Developed by Monish Raj T, B.Tech in Artificial Intelligence and Data Science at Karpagam College of Engineering.

## License

This project is open-source and available for modification.
