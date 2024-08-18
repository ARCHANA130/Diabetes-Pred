import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Load the diabetes dataset
diabetes_df = pd.read_csv("Diabetes.csv")

# Split the dataset into features and target
X = diabetes_df.drop("Outcome", axis=1)
y = diabetes_df["Outcome"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Create the Streamlit app
def main():
    st.title("Diabetes Prediction App")
    st.write("Enter the following information to predict diabetes:")

    # Create input fields for user to enter information
    id = st.text_input("ID")
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=0)
    insulin = st.number_input("Insulin Level", min_value=0, max_value=1000, value=0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=0.0, step=0.1)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.0, step=0.001)
   # diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0, max_value=3, value=0, step=0.001)
    age = st.number_input("Age", min_value=0, max_value=120, value=0)

    # Create a button to trigger the prediction
    if st.button("Predict"):
        # Create a dataframe with the user input
        user_data = pd.DataFrame(
            {   "Id":[id],
                "Pregnancies": [pregnancies],
                "Glucose": [glucose],
                "BloodPressure": [blood_pressure],
                "SkinThickness": [skin_thickness],
                "Insulin": [insulin],
                "BMI": [bmi],
                "DiabetesPedigreeFunction": [diabetes_pedigree],
                "Age": [age],
            }
        )

        # Make the prediction
        prediction = clf.predict(user_data)

        # Display the prediction
        if prediction[0] == 0:
            st.write("You are not diabetic.")
        else:
            st.write("You are diabetic.")

if __name__ == "__main__":
    main()