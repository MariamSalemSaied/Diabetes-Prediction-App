import streamlit as st
import pandas as pd
import joblib
from typeguard import value

def show():
    st.title("Model")


    model = joblib.load("svm_rbf_model.pkl")
    scaler = joblib.load("scaler_fs.pkl")
    scaler_full = joblib.load("scaler_full.pkl")
    selected_features = joblib.load("selected_features.pkl")

    def yes_no_to_int(value):
        return 1 if value == "Yes" else 0
    
    def sex_to_int(value):
        return 1 if value == "Male" else 0
    
    def genhlth_to_int(value):
        return 1 if value == "Excellent" else 2 if value == "Very good" else 3 if value == "Good" else 4 if value == "Fair" else 5

    def age_map(value):
        if value < 25:
            return 1
        elif value < 30:
            return 2
        elif value < 35:
            return 3
        elif value < 40:
            return 4
        elif value < 45:
            return 5
        elif value < 50:
            return 6
        elif value < 55:
            return 7
        elif value < 60:
            return 8
        elif value < 65:
            return 9
        elif value < 70:
            return 10
        elif value < 75:
            return 11
        elif value < 80:
            return 12
        elif value >= 80:
            return 13

    HighBP = st.selectbox("High Blood Pressure", ["Yes","No"])
    HighChol = st.selectbox("High Cholesterol", ["Yes","No"])
    CholCheck = st.selectbox("Cholesterol Check in Last 5 Years", ["Yes","No"])
    BMI = st.number_input("BMI", min_value=10.0, max_value=80.0, value=25.0)
    Stroke = st.selectbox("Ever had a Stroke", ["Yes","No"])
    HeartDiseaseorAttack = st.selectbox("Heart Disease or Attack", ["Yes","No"])
    HvyAlcoholConsump = st.selectbox("Heavy Alcohol Consumption", ["Yes","No"])
    GenHlth = st.selectbox("General Health", ["Excellent", "Very good", "Good", "Fair", "Poor"])
    Sex = st.selectbox("Sex", ["Female", "Male"])
    Age = st.number_input("Age", min_value=18, max_value=80, value=30)
  
    user_input = {
        "HighBP": yes_no_to_int(HighBP),
        "HighChol": yes_no_to_int(HighChol),
        "CholCheck": yes_no_to_int(CholCheck),
        "BMI": BMI,
        "Stroke": yes_no_to_int(Stroke),
        "HeartDiseaseorAttack": yes_no_to_int(HeartDiseaseorAttack),
        "GenHlth": genhlth_to_int(GenHlth),
        "Age": age_map(Age),
        "HvyAlcoholConsump": yes_no_to_int(HvyAlcoholConsump),
        "Sex": sex_to_int(Sex),
    }

    input_df = pd.DataFrame([user_input])

    # ensure correct order
    input_df = input_df[selected_features]

    if st.button("Predict"):

        # create full 21-feature input
        all_features = scaler_full.feature_names_in_
        full_input = pd.DataFrame([{col: 0 for col in all_features}])
        # fill user values
        for col, value in user_input.items():
            full_input[col] = value

        # scale all 21 features
        full_scaled = scaler_full.transform(full_input)
        full_scaled_df = pd.DataFrame(full_scaled, columns=all_features)

        #select the 10 features
        selected_scaled = full_scaled_df[selected_features]

        #second scaling
        input_scaled = scaler.transform(selected_scaled)

        #predict
        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
            st.header("Prediction: Diabetes")
        else:
            st.header("Prediction: No Diabetes")


        
    

        