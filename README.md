# Diabetes Prediction App

A Machine Learning and Deep Learning web application that predicts the risk of diabetes using patient health information. The application is built with **TensorFlow**, **Scikit-learn**, and **Streamlit**, providing an interactive and user-friendly interface for prediction.

---

## Features

* Predicts diabetes risk using a trained Artificial Neural Network (ANN).
* Clean and responsive Streamlit user interface.
* Three-column layout for better user experience.
* Displays prediction probability.
* Uses MinMaxScaler for feature scaling.
* Supports 21 health-related input features.
* Fast and easy to use.

---

## Dataset Features

The model uses the following input features:

* High Blood Pressure
* High Cholesterol
* Cholesterol Check
* BMI
* Smoker
* Stroke
* Heart Disease or Heart Attack
* Physical Activity
* Fruits
* Vegetables
* Heavy Alcohol Consumption
* Any Healthcare
* No Doctor Because of Cost
* General Health
* Mental Health
* Physical Health
* Difficulty Walking
* Gender
* Age
* Education
* Income

---

## Technologies Used

* Python
* Streamlit
* TensorFlow / Keras
* Scikit-learn
* Pandas
* NumPy
* Joblib

---

## Project Structure

```
Diabetes-Prediction-App/
│── app.py
│── diabetes_model.keras
│── scaler.pkl
│── requirements.txt
│── README.md
│── diabetes.csv
│── Diabetes_Prediction.ipynb
└── .gitignore
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Diabetes-Prediction-App.git
```

### 2. Navigate to the Project Folder

```bash
cd Diabetes-Prediction-App
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

---

## Model

The application uses an **Artificial Neural Network (ANN)** built with TensorFlow/Keras.

### Workflow

1. Load the trained model.
2. Load the saved MinMaxScaler.
3. Accept user inputs.
4. Scale the input data.
5. Predict diabetes probability.
6. Display the prediction result.

---

## Prediction Output

* Low Risk of Diabetes
* High Risk of Diabetes

The application also displays the prediction probability.

---

## Requirements

* streamlit
* tensorflow
* pandas
* numpy
* scikit-learn
* joblib

Install all packages using:

```bash
pip install -r requirements.txt
```

---

## Application Preview

You can add video of the application here after deployment.

---

## Author

**Putnala Sridhar**

---

## License

This project is developed for educational and learning purposes.
