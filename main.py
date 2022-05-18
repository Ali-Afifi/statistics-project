import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# cleaning the data

df = pd.read_csv("./diabetes.csv") # having_diabetes = 1 , not_having_diabetes = 0
df["gender"] = (df["gender"] == "male").astype(int)
df["diabetes"] = (df["diabetes"] == "Diabetes").astype(int)  # male = 1 , female = 0
df.drop(["patient_number"], axis=1, inplace=True)
df["bmi"] = df["bmi"].str.replace(",", ".").astype(float)
df["waist_hip_ratio"] = df["waist_hip_ratio"].str.replace(",", ".").astype(float)
df["chol_hdl_ratio"] = df["chol_hdl_ratio"].str.replace(",", ".").astype(float)
df_class_true = df[df["diabetes"] == 1]
df_class_false = df[df["diabetes"] == 0]
df_class_false = df_class_false.sample(df_class_true["diabetes"].count(), random_state=42)
df = pd.concat([df_class_false, df_class_true], axis=0)
trainset, testset = train_test_split(df, test_size=0.25, random_state=42)
X_train = trainset.drop(["diabetes"], axis=1)
y_train = trainset["diabetes"]
X_test = testset.drop(["diabetes"], axis=1)
y_test = testset["diabetes"]


# traning the model

error_rate = []
for i in range(1, 89):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

req_k_value = error_rate.index(min(error_rate))+1

classifier = KNeighborsClassifier(n_neighbors=40)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_pred, y_test)*100


# the web page


st.title("Predicting Diabetes")
st.write("### Please Enter your data")

gender = st.selectbox("Select Gender", ("male", "female"))
age = st.select_slider("Age", options=range(19, 93), value=21)
height = st.select_slider("Height in inches", options=range(52, 76), value=65)
weight = st.select_slider("Weight in pounds", options=range(99, 325), value=250)
cholesterol = st.select_slider("Your Cholesterol Level", options=range(78, 443), value=170)
glucose = st.select_slider("Your Glucose Level", options=range(48, 385), value=125)
hdl_chol = st.select_slider("Your HDL Level", options=range(12, 120), value=40)
systolic_bp = st.select_slider("Systolic blood pressure", options=range(90,250), value=110)
diastolic_bp = st.select_slider("Diastolic blood pressure", options=range(48,124), value=70)
waist = st.select_slider("Waist circumfrance in inches", options=range(26,56), value=37)
hip = st.select_slider("Waist circumfrance in inches", options=range(30,64), value=40)
bmi = (703 * weight) / (height**2)
chol_hdl_ratio = cholesterol / hdl_chol
waist_hip_ratio = waist / hip
gender_num = 1 if gender=="male" else 0 

button = st.button("Predict")


if (button):
    my_data = {
        "cholesterol": cholesterol,
        "glucose": glucose,
        "hdl_chol": hdl_chol,
        "chol_hdl_ratio": chol_hdl_ratio,
        "age": age,
        "gender": gender_num,
        "height": height,
        "weight": weight,
        "bmi": bmi,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "waist": waist,
        "hip": hip,
        "waist_hip_ratio": waist_hip_ratio
    }
    my_df = pd.DataFrame(data=my_data, index=[0])
    prediction = classifier.predict(my_df)[0]
    if (prediction):
        st.write("#### You have diabetes")
    else:
        st.write("#### You do not have diabetes")

    st.write(f"##### prediction accuracy: {round(accuracy, 2)}%")
