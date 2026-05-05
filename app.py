import streamlit as st
import EDA
import Model

st.title("Diabetes prediction app")

#background color
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(45deg, #90AB8B, #5A7863, #3B4953);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }

    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    </style>
    """,
    unsafe_allow_html=True
    )



st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select page", ["EDA", "Model"])

if page == "EDA" :
    EDA.show()
elif page == "Model" :
    Model.show()
