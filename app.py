# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# LOAD DATA
df = pd.read_csv("diabetes_dataset.csv")



# ENCODE CATEGORICAL DATA
df = pd.get_dummies(
    df,
    columns=["gender", "smoking_history"],
    drop_first=True
)



# HEADINGS
st.title("Diabetes Checkup – Clinical Decision Support")


# SPLIT DATA
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)



# USER INPUT FUNCTION
def user_report():
    st.header("Patient Data Form")
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", 1, 120, 33)
            bmi = st.number_input("BMI", 10.0, 60.0, 22.0)
            hba1c = st.number_input("HbA1c Level", 3.0, 15.0, 5.5)
        
        with col2:
            glucose = st.number_input("Blood Glucose Level", 50, 400, 110)
            gender = st.selectbox("Gender", ["Male", "Female"])
            smoking = st.selectbox(
                "Smoking History",
                ["never", "former", "current", "not current", "ever", "No Info"]
            )
        
        col3, col4 = st.columns(2)
        with col3:
            hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        with col4:
            heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        
        submitted = st.form_submit_button("Generate Report", use_container_width=True)
    
    if not submitted:
        st.info("👈 Fill out the form and click 'Generate Report'")
        return None

    data = {
        "age": age,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose,
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "gender_" + gender: 1,
        "smoking_history_" + smoking: 1
    }

    user_df = pd.DataFrame(data, index=[0])
    return user_df



# GET USER DATA
user_data = user_report()

if user_data is None:
    st.stop()

# align columns with training data
user_data = user_data.reindex(
    columns=x_train.columns,
    fill_value=0
)

# TRAIN MODEL
rf = RandomForestClassifier(random_state=0)
rf.fit(x_train, y_train)

user_result = rf.predict(user_data)
user_prob = rf.predict_proba(user_data)[0][1]

# Calculate accuracy
accuracy = accuracy_score(y_test, rf.predict(x_test)) * 100
color = "red" if user_result[0] == 1 else "blue"

# CREATE TABS
tab1, tab2, tab3 = st.tabs(["🩺 Risk Assessment", "📊 Analysis", "📈 Visualizations"])

with tab1:
    st.subheader("Risk Assessment Results")
    
    col_risk1, col_risk2 = st.columns(2)
    
    with col_risk1:
        if user_result[0] == 1:
            st.error(f"⚠️ High Diabetes Risk Detected")
            st.metric("Risk Score", f"{user_prob:.2%}")
        else:
            st.success(f"✅ Low Diabetes Risk Detected")
            st.metric("Risk Score", f"{user_prob:.2%}")
    
    with col_risk2:
        st.metric("Model Accuracy", f"{accuracy:.2f}%")
        
        if user_result[0] == 0:
            st.success("You are not Diabetic")
        else:
            st.error("You are Diabetic")
    
    st.divider()
    st.subheader("Patient Data Summary")
    
    col_data1, col_data2, col_data3 = st.columns(3)
    
    with col_data1:
        st.metric("Age", int(user_data["age"].values[0]))
        st.metric("BMI", f"{user_data['bmi'].values[0]:.1f}")
    
    with col_data2:
        st.metric("HbA1c Level", f"{user_data['HbA1c_level'].values[0]:.1f}%")
        st.metric("Blood Glucose", f"{user_data['blood_glucose_level'].values[0]:.0f} mg/dL")
    
    with col_data3:
        hypertension_status = "Yes" if user_data['hypertension'].values[0] == 1 else "No"
        heart_disease_status = "Yes" if user_data['heart_disease'].values[0] == 1 else "No"
        st.metric("Hypertension", hypertension_status)
        st.metric("Heart Disease", heart_disease_status)

with tab2:
    st.subheader("Feature Importance")
    st.write("Factors that contributed most to this risk prediction:")
    
    importance_df = pd.DataFrame({
        "feature": x_train.columns,
        "contribution": rf.feature_importances_
    }).sort_values("contribution")

    fig3 = plt.figure(figsize=(8, 5))
    plt.barh(
        importance_df.tail(6)["feature"],
        importance_df.tail(6)["contribution"]
    )
    plt.xlabel("Relative Contribution to Risk Prediction")
    plt.ylabel("Feature")
    st.pyplot(fig3)

    st.caption(
        "Feature importance reflects global model behavior and is intended to support clinical interpretation, not causal inference."
    )

with tab3:
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        st.subheader("HbA1c vs Blood Glucose (Population Context)")
        fig1 = plt.figure()
        sns.scatterplot(
            x="HbA1c_level",
            y="blood_glucose_level",
            data=df,
            hue="diabetes",
            palette="coolwarm",
            alpha=0.5
        )
        plt.scatter(
            user_data["HbA1c_level"],
            user_data["blood_glucose_level"],
            color=color,
            s=150,
            label="Patient"
        )
        plt.legend()
        st.pyplot(fig1)
    
    with col_viz2:
        st.subheader("BMI vs Predicted Risk")
        
        bmi_range = np.linspace(15, 45, 50)
        temp = user_data.copy()
        
        risks = []
        for bmi_val in bmi_range:
            temp["bmi"] = bmi_val
            risks.append(rf.predict_proba(temp)[0][1])
        
        fig2 = plt.figure()
        plt.plot(bmi_range, risks)
        plt.scatter(user_data["bmi"], user_prob, color=color, s=100)
        plt.xlabel("BMI")
        plt.ylabel("Predicted Diabetes Risk")
        st.pyplot(fig2)

