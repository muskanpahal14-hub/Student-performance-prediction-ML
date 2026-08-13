import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Page settings
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Prediction")
st.write("Predict whether a student is likely to Pass or Fail.")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("student_performance_dataset.csv")

try:
    df = load_data()

    # Features and target
    features = [
        "Study_Hours",
        "Attendance",
        "Assignments_Completed",
        "Previous_Exam_Score"
    ]

    target = "Pass"

    # Check required columns
    required_columns = features + [target]

    if not all(column in df.columns for column in required_columns):
        st.error(
            "Dataset columns do not match the required format. "
            "Please check the CSV file."
        )
        st.write("Required columns:", required_columns)
        st.stop()

    X = df[features]
    y = df[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Standardization
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression model
    model = LogisticRegression(random_state=42)

    model.fit(X_train_scaled, y_train)

    # Model accuracy
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    st.success(f"Model Accuracy: {accuracy * 100:.2f}%")

    st.subheader("Enter Student Details")

    study_hours = st.number_input(
        "Study Hours",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.5
    )

    attendance = st.number_input(
        "Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=85.0,
        step=1.0
    )

    assignments = st.number_input(
        "Assignments Completed",
        min_value=0,
        max_value=100,
        value=8,
        step=1
    )

    previous_score = st.number_input(
        "Previous Exam Score",
        min_value=0.0,
        max_value=100.0,
        value=72.0,
        step=1.0
    )

    if st.button("🔮 Predict Performance"):

        input_data = pd.DataFrame({
            "Study_Hours": [study_hours],
            "Attendance": [attendance],
            "Assignments_Completed": [assignments],
            "Previous_Exam_Score": [previous_score]
        })

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]

        if prediction == 1:
            st.success("🎉 Prediction: PASS")
        else:
            st.error("📚 Prediction: FAIL")

        st.write(
            f"Probability of Pass: **{probabilities[1] * 100:.2f}%**"
        )

        st.write(
            f"Probability of Fail: **{probabilities[0] * 100:.2f}%**"
        )

except FileNotFoundError:
    st.error("Dataset file not found.")

    st.info(
        "Please upload 'student_performance_dataset.csv' "
        "in the same GitHub repository as app.py."
    )
