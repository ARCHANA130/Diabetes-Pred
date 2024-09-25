import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Load the diabetes dataset
diabetes_df = pd.read_csv("Diabetes.csv")

# Split the dataset into features and target

# Add custom CSS to make the background more pretty


PAGE_CONFIG = {
    "page_title": "MyApp",
    "layout": "centered",
    "initial_sidebar_state": "auto"
}
st.set_page_config(**PAGE_CONFIG)
# Add custom CSS to make the background more pretty
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://plus.unsplash.com/premium_photo-1668104452882-2ae0969bb2cc?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)




X = diabetes_df.drop("Outcome", axis=1)
y = diabetes_df["Outcome"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Create the Streamlit app

def main():
    st.title("HerHealth")
    st.write("Enter the following information to predict diabetes:")

    # Create input fields for user to enter information
    
    id = st.text_input("**ID**")
    # Add custom CSS to style the number inputs
   
    st.markdown(
    """
    <style>
    /* Style the input field */
    input[type=number] {
        background-color: 	#ffecf2;
        color: #333333;
        font-size: 18px;
        padding: 10px;
        border-radius: 5px;
        border: 2px solid #ccc;
    }

 
    </style>
    """,
    unsafe_allow_html=True
)
    pregnancies = st.number_input("**Number of Pregnancies**", min_value=0, max_value=20, value=0)
    glucose = st.number_input("**Glucose Level**", min_value=0, max_value=200, value=0)
    blood_pressure = st.number_input("**Blood Pressure**", min_value=0, max_value=200, value=0)
    skin_thickness = st.number_input("**Skin Thickness**", min_value=0, max_value=100, value=0)
    insulin = st.number_input("**Insulin Level**", min_value=0, max_value=1000, value=0)
    bmi = st.number_input("**BMI**", min_value=0.0, max_value=60.0, value=0.0, step=0.1)
    diabetes_pedigree = st.number_input("**Diabetes Pedigree Function**", min_value=0.0, max_value=3.0, value=0.0, step=0.001)
    age = st.number_input("**Age**", min_value=0, max_value=120, value=0)


    # Create a button to trigger the prediction
    # Add custom CSS to style the predict button
    predict_button_style = """
    <style>
    div.stButton > button:first-child {
        background-color:  #9F2B68;
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius:6px;
        cursor: pointer;
        font-size: 16px;
        
    }
    div.stButton > button:first-child:hover {
        background-color: #DE3163;
    }
    </style>
    """
    st.markdown(predict_button_style, unsafe_allow_html=True)

    if st.button("**Predict**"):
        
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
        # Display the prediction in a beautiful box
        if prediction[0] == 0:
            st.success("**You are not diabetic.**")
            st.markdown(
            """
            <div style="background-color: #ffecf2; padding: 10px; border-radius: 5px;">
                <h4>Here are some suggestions to prevent diabetes:</h4>
                <ul>
                <li>Maintain a healthy weight.</li>
                <li>Eat a balanced diet rich in fruits, vegetables, and whole grains.</li>
                <li>Exercise regularly.</li>
                <li>Avoid sugary drinks and foods high in saturated fats.</li>
                <li>Don't smoke.</li>
                <li>Monitor your blood sugar levels if you are at risk.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True
            )
        else:
            st.error("**You are diabetic.**")
            st.markdown(
            """
            <div style="background-color: #ffecf2; padding: 10px; border-radius: 5px;">
                <h4>Here are some treatments for diabetes:</h4>
                <ul>
                <li>Monitor your blood sugar levels regularly.</li>
                <li>Take prescribed medications as directed by your healthcare provider.</li>
                <li>Follow a healthy eating plan and monitor carbohydrate intake.</li>
                <li>Engage in regular physical activity.</li>
                <li>Maintain a healthy weight.</li>
                <li>Attend regular check-ups with your healthcare provider.</li>
                <li>Consider insulin therapy if recommended by your healthcare provider.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True
            )

        
      
       

if __name__ == "__main__":
    main()