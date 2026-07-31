# Library Imports

import joblib
import streamlit as st
import pandas as pd
import os


# Title Organizing 

st.set_page_config(page_title = "Employee Salary Prediction System",
page_icon = "💼", layout = "centered")

st.markdown("""
<style>
/* Main Background */
.stapp{
    background-color:#0E1117:
    color:white;
} 

/* Sidebar */
section[data-testid = "stSidebar"] 
{ Background-color:#1B1F2A;
}

section[data-testid="stSidebar"]
*{
    color: white !important;
}

/* Buttons */ 
.stButton > button{background-color:#00B4D8;
color:white;
border-radius:10px;
font-size:18x;
font-weight;bold;
height:50px;
border:none
}

.stButton > button:hover { background-color: #0096C7;
}

/* Input Boxes */
input{border-radius:10px !important;
}
</style>
""" , unsafe_allow_html =True )



# BASE_DIR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Model Paths

model_path = os.path.join(BASE_DIR, "models/model.pkl")
scaler_path = os.path.join(BASE_DIR, "models/scaler.pkl")
feature_path = os.path.join(BASE_DIR, "models/feature.pkl")

# Load 

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature = joblib.load(feature_path)


# Header

st.sidebar.title("ℹ️ About Project") 

st.sidebar.markdown(""" ### 🤖 Machine Learning Model

**Algorithm:** Logistic Regression

**Dataset:** Adult Income Dataset

**Language:** Python

**Framework:** Streamlit

**Libraries:** Numpy
Pandas
 Scikit-learn
 Joblib
 Streamlit
 """)

# Title

st.title("💼 EMPLOYEE SALARY PREDICTION SYSTEM")
st.markdown(""" ### Predict Whether an Employee Earns More Than $50K Annually.

This Application Uses a Machine Learning Logistic Regression Model""")

st.divider()

# Collecting User Input

st.subheader("👤 Employee Information")

col1, col2 = st.columns(2) # >= Two Columns Layout

with col1:

    age = st.number_input ("Age",min_value = 18,max_value = 100)
    fnlwgt = st.number_input("Final Weight (fnlwgt)")
    workclass = st.selectbox("Workclass",["Private","Self-emp-not-inc","Local-gov","State-gov","Self-emp-inc","Federal-gov","Without-pay","Never-worked"])

    education = st.selectbox("Education",["HS-grad","Some-college","Bachelors","Masters","Assoc-voc","11th","Assoc-acdm","10th","7th-8th","Prof-school","9th","12th",
"Doctorate","5th-6th","1st-4th","Preschool"])

    education_num = st.number_input("Education Number")
    marital_status = st.selectbox("Marital Status",["Married-civ-spouse","Never-married","Divorced","Separated","Widowed","Married-spouse-absent","Married-AF-spouse"])

    occupation = st.selectbox("Occupation",["Prof-specialty","Craft-repair","Exec-managerial","Adm-clerical","Sales","Machine-op-inspct","Transport-moving","Handlers-cleaners",
"Farming-fishing","Tech-support","Protective-serv","Prive-house-serve","Armed-Forces","Other-service"])

with col2:

    relationship = st.selectbox("Relationship",["Wife","Own-child","Husband","Not-in-family","Unmarried","Other-relative"])

    race = st.selectbox("Race",["White","Black","Asian-Pac-Islander","Amer-Indian-Eskimo","Other"])

    sex = st.selectbox("Sex",["Male","Female"])
    capital_gain = st.number_input("Capital Gain")
    capital_loss = st.number_input("Capital Loss")
    hours_per_week = st.number_input("Hours Per Week")

    native_country = st.selectbox("Native Country",["United-States","Mexico","Philippines","Germany","Canada","Puerto-Rico","El-Salvador","India","Cuba","England","Jamaica",
"South","China","Italy","Dominican-Republic","Vietnam","Guatemala","Japan","Poland","Columbia","Taiwan","Haiti","Iran","Portugal","Nicaragua","Peru","Greece","France",
"Ecuador","Ireland","Hong","Trinadad&Tobago","Cambodia","Laos","Thailand","Yugoslavia","Outlying-Us(Guam-USVI-etc)","Honduras","Hungary","Scotland","Holand-Netherland"])
    
# UI Creation Successfully Completed + Collected User Info


# Creating Button For Prediction

predict_btn = st.button("🚀 Predict Salary",use_container_width = True) # >= BUtton

if predict_btn:

    # Processing User Input

    with st.spinner("🤖 AI is Analyzing Employee Proflie..."):

        # Collecting Input Values
        
        input_data = [age,fnlwgt,workclass,education,education_num,marital_status,occupation,relationship,race, # >= Collected User Info
sex,capital_gain,capital_loss,hours_per_week,native_country]

        # converting Input Into DataFrame

        input_df = pd.DataFrame([input_data],columns = ["age","fnlwgt","workclass","education","education.num","marital.status",
"occupation","relationship","race","sex","capital.gain","capital.loss","hours.per.week","native.country"])  

        # Encoding Categorical Features

        input_df = pd.get_dummies(input_df) 

        # Match Training Features
    
        features  = joblib.load("models/feature.pkl")            # >= Feature alignment For Fill Missing columns in(0)
        input_df = input_df.reindex(columns=features,fill_value=0)

        print(input_df)

        # Scaling The Data Like In Training

        input_df = scaler.transform(input_df) 

        #  Final Prediction

        prediction = model.predict(input_df) 

        st.subheader("📊 Prediction Result")

        if prediction[0]==">50K":
         st.success(""" ### ✅ Prediction: Above $50000
         
         The Employee is Predicted to Earn **More Than $50000 Per Year.**
         """)

        else:
            st.warning(""" ### ⚠️ Prediction $50000 or Below
            The Employee is Predicted To Earn **$50000 or Less Per Year.** 
            """)
        st.divider()


# Footer

st.markdown(""" <div style = "text-align:center;
color:gray;">
        
Built with ❤️ using <b>Python</b>
,<b>Streamlit</b>
,<b>Numpy</b>
,<b>Pandas</b> and 
<b> Scikit-learn</b>
        
<br><br>

Developed by <b> HABEEB KHAN </b>

</div>
""" , unsafe_allow_html = True)
      


        
        

